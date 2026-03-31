"""
Export EdgePoint2 E64 to ONNX for Axelera Metis M2 compilation.

Metis M2 AIPU compiler constraints addressed here:
  1. pass_pword_pad: all conv IC and OC must be multiples of 64.
  2. tile_dwpu:      DWPU ops (AveragePool, Resize, grouped-conv) need C % 64 == 0.
  3. Slice support:  only single-axis, stride-1 slices; no strided spatial slices.

Strategy:
  - Zero-pad single-channel input to 64ch via a Pad ONNX node.
  - Rebuild all backbone convolutions with weights zero-padded to 64ch boundaries.
    Zero-padded IC/OC contribute nothing to outputs (matching weight columns = 0).
  - Use F.avg_pool2d for spatial decimation (64ch is fine for the DWPU).
  - Descriptor branches: upsample at 64ch (DWPU-safe), then concatenate →192ch.
    The desc_head entry 1×1 conv weight is remapped for the non-contiguous
    real-channel layout [_x2:ch0-15, pad:ch16-63, x2:ch64-111, pad:ch112-127, x3:ch128-191].
    Subsequent desc_head layers are zero-padded from 128→192ch normally.
  - Detection path: conv1/2/3 and score_head rebuilt with 64ch internals.
  - PixelShuffle(2) excluded — CPU decoder handles this.
  - InstanceNorm2d excluded — torch-totensor maps images to [0,1].

Exported outputs:
  x1_feat:       [1, 64,  H/2,  W/2 ] padded backbone feature map
  x2_feat:       [1, 64,  H/8,  W/8 ] padded backbone feature map
  x3_feat:       [1, 64,  H/32, W/32] padded backbone feature map

The multi-scale descriptor and detector heads are reconstructed on the CPU in
the decoder. This mirrors the successful SuperPoint deployment strategy and
avoids the Metis midend fusion failures around Resize/Concat/Add-heavy regions.
The exporter also writes a compact `edgepoint_heads.bin` blob containing the
CPU-side head weights for the GStreamer decoder.

Usage (from voyager-sdk root or anywhere):
  python ax_models/custom/edgepoint/export_onnx.py \\
      --weights ax_models/custom/edgepoint/weights/E64.pth \\
      --output  ax_models/custom/edgepoint/weights/edgepoint_encoder.onnx \\
      --edgepoint-repo /home/Vydar/EdgePoint2 \\
      --height 512 --width 512
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 64-channel padding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pad_conv(conv: nn.Conv2d, in_c: int, out_c: int) -> nn.Conv2d:
    """Return a new Conv2d(in_c, out_c) with pretrained weights zero-padded."""
    new = nn.Conv2d(
        in_c, out_c, conv.kernel_size, conv.stride, conv.padding,
        dilation=conv.dilation, bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.zero_()
        r_out = min(conv.out_channels, out_c)
        r_in  = min(conv.in_channels,  in_c)
        new.weight[:r_out, :r_in] = conv.weight[:r_out, :r_in]
        if conv.bias is not None:
            new.bias.zero_()
            new.bias[:r_out] = conv.bias[:r_out]
    return new


def _pad_bn(bn: nn.BatchNorm2d, c: int) -> nn.BatchNorm2d:
    """Return a new BatchNorm2d(c) with stats zero-padded from bn."""
    new = nn.BatchNorm2d(c, eps=bn.eps, momentum=bn.momentum)
    with torch.no_grad():
        r = bn.num_features
        new.weight.zero_();       new.weight[:r]      = bn.weight
        new.bias.zero_();         new.bias[:r]        = bn.bias
        new.running_mean.zero_(); new.running_mean[:r] = bn.running_mean
        # Padded channels always carry 0; var=1 avoids division-by-zero in BN.
        new.running_var.fill_(1.0); new.running_var[:r] = bn.running_var
    return new


def _pad_basic_layer(bl, in_c: int, out_c: int) -> nn.Sequential:
    """Rebuild a BasicLayer (Conv → BN → ReLU) with padded channel counts."""
    return nn.Sequential(
        _pad_conv(bl.layer[0], in_c, out_c),
        _pad_bn(bl.layer[1],   out_c),
        bl.layer[2],   # ReLU — stateless, safe to share
    )


class _PaddedResNetLayer(nn.Module):
    """ResNetLayer rebuilt with zero-padded (in_c → out_c) channel counts."""

    def __init__(self, rl, in_c: int, out_c: int):
        super().__init__()
        # rl.layer: Conv → BN → ReLU → Conv → BN
        self.conv1 = _pad_conv(rl.layer[0], in_c,  out_c)
        self.bn1   = _pad_bn(rl.layer[1],   out_c)
        self.relu1 = rl.layer[2]
        self.conv2 = _pad_conv(rl.layer[3], out_c, out_c)
        self.bn2   = _pad_bn(rl.layer[4],   out_c)
        self.skip  = (nn.Identity() if isinstance(rl.skip, nn.Identity)
                      else _pad_conv(rl.skip, in_c, out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x))


def _make_dense_gconv(grouped_conv: nn.Conv2d) -> nn.Conv2d:
    """Convert grouped Conv2d → dense Conv2d with block-diagonal weights.

    Mathematically identical to the original.  Avoids the tile_dwpu bug when
    C/G (channels per group) is below the DWPU minimum.
    """
    G  = grouped_conv.groups
    C  = grouped_conv.out_channels
    CG = C // G
    conv = nn.Conv2d(C, C, grouped_conv.kernel_size, grouped_conv.stride,
                     grouped_conv.padding, bias=grouped_conv.bias is not None)
    with torch.no_grad():
        conv.weight.zero_()
        for g in range(G):
            sl = slice(g * CG, (g + 1) * CG)
            conv.weight[sl, sl] = grouped_conv.weight[sl]
        if grouped_conv.bias is not None:
            conv.bias.copy_(grouped_conv.bias)
    return conv


def _remap_desc_pw1(orig_conv: nn.Conv2d) -> nn.Conv2d:
    """Build Conv2d(192, 192, 1) from the original Conv2d(128, 128, 1).

    After concatenating three 64-padded identity branches, the 192-channel
    tensor has this layout (only 128 channels carry real data):

      ch   0– 15: real  (_x2  → original input ch  0–15)
      ch  16– 63: zero  (padding of _x2 from 16→64ch)
      ch  64–111: real  (x2   → original input ch 16–63)
      ch 112–127: zero  (padding of x2 from 48→64ch)
      ch 128–191: real  (x3   → original input ch 64–127)

    Weight columns are remapped accordingly so the arithmetic is identical
    to the original 128-channel convolution.  Output channels 128–191 are
    zero-padded (next layer only uses first 128 real outputs).
    """
    assert orig_conv.kernel_size == (1, 1), "expected 1×1 pointwise conv"
    new = nn.Conv2d(192, 192, 1, bias=orig_conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        # IC remap: scatter original columns to their new positions
        new.weight[:128,   0: 16, 0, 0] = orig_conv.weight[:,  0:16,  0, 0]
        new.weight[:128,  64:112, 0, 0] = orig_conv.weight[:, 16:64,  0, 0]
        new.weight[:128, 128:192, 0, 0] = orig_conv.weight[:, 64:128, 0, 0]
        # OC 128–191 remain zero (padding)
        if orig_conv.bias is not None:
            new.bias.zero_()
            new.bias[:128] = orig_conv.bias
    return new


# ─────────────────────────────────────────────────────────────────────────────

def load_model(edgepoint_repo: str, weights_path: str):
    """Load EdgePoint2 E64 from the cloned repo + a .pth weights file."""
    sys.path.insert(0, str(edgepoint_repo))
    from model.model import EdgePoint2

    cfg = {'c1': 16, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 64, 'cdetect': 16}
    base = EdgePoint2(**cfg)
    state = torch.load(weights_path, map_location='cpu')
    base.load_state_dict(state, strict=True)
    base.eval()
    return base


def _materialize_resize_constants_as_initializers(output_path: str) -> None:
    """Rewrite Resize helper Constant nodes into graph initializers.

    TVM imports Constant-producing subgraphs feeding Resize as regular Relay
    expressions, which can make the Metis partitioner count them as extra
    non-constant region inputs. Rewriting them into initializers keeps the ONNX
    semantics identical while presenting Resize as one runtime tensor plus
    compile-time constant metadata.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(output_path)
    graph = model.graph

    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    constant_values = {}
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value_attr = next((attr for attr in node.attribute if attr.name == "value"), None)
        if value_attr is None:
            continue
        constant_values[node.output[0]] = numpy_helper.to_array(
            onnx.helper.get_attribute_value(value_attr)
        )

    outputs_to_remove = set()
    existing_inits = {init.name for init in graph.initializer}

    for node in graph.node:
        if node.op_type != "Resize":
            continue
        for idx in (1, 2):
            src = node.input[idx]
            if src not in constant_values:
                continue
            if len(consumers.get(src, [])) != 1:
                continue

            init_name = f"{src.replace('/', '_').strip('_')}_init"
            while init_name in existing_inits:
                init_name += "_"
            tensor = numpy_helper.from_array(constant_values[src], name=init_name)
            graph.initializer.append(tensor)
            existing_inits.add(init_name)
            node.input[idx] = init_name

            for producer in graph.node:
                if producer.op_type == "Constant" and producer.output and producer.output[0] == src:
                    outputs_to_remove.add(src)
                    break

    if outputs_to_remove:
        kept_nodes = [
            node for node in graph.node
            if not (node.op_type == "Constant" and node.output and node.output[0] in outputs_to_remove)
        ]
        del graph.node[:]
        graph.node.extend(kept_nodes)
        onnx.save(model, output_path)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fold BatchNorm parameters into a Conv2d for inference export."""
    weight = conv.weight.detach().cpu().float()
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, dtype=torch.float32)
    else:
        bias = conv.bias.detach().cpu().float()

    gamma = bn.weight.detach().cpu().float()
    beta = bn.bias.detach().cpu().float()
    mean = bn.running_mean.detach().cpu().float()
    var = bn.running_var.detach().cpu().float()
    inv_std = gamma / torch.sqrt(var + bn.eps)

    reshape_dims = [conv.out_channels] + [1] * (weight.ndim - 1)
    fused_weight = weight * inv_std.view(*reshape_dims)
    fused_bias = beta + (bias - mean) * inv_std
    return fused_weight.numpy(), fused_bias.numpy()


def _write_tensor_blob(handle, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=np.float32, order='C')
    handle.write(struct.pack('<I', arr.ndim))
    handle.write(struct.pack('<' + 'I' * arr.ndim, *arr.shape))
    handle.write(arr.tobytes())


def _export_head_weights(base: nn.Module, weights_blob_path: str) -> None:
    """Export the CPU-side EdgePoint2 heads to a compact binary blob."""
    Path(weights_blob_path).parent.mkdir(parents=True, exist_ok=True)

    desc_dw_w, desc_dw_b = _fuse_conv_bn(
        base.desc_head[1].layer[0],
        base.desc_head[1].layer[1],
    )

    tensors = [
        base.conv1.weight.detach().cpu().numpy(),
        base.conv1.bias.detach().cpu().numpy(),
        base.conv2.weight.detach().cpu().numpy(),
        base.conv2.bias.detach().cpu().numpy(),
        base.conv3.weight.detach().cpu().numpy(),
        base.conv3.bias.detach().cpu().numpy(),
        base.score_head[0].weight.detach().cpu().numpy(),
        base.score_head[0].bias.detach().cpu().numpy(),
        base.score_head[2].weight.detach().cpu().numpy(),
        base.score_head[2].bias.detach().cpu().numpy(),
        base.score_head[4].weight.detach().cpu().numpy(),
        base.score_head[4].bias.detach().cpu().numpy(),
        base.desc_head[0].weight.detach().cpu().numpy(),
        base.desc_head[0].bias.detach().cpu().numpy(),
        desc_dw_w,
        desc_dw_b,
        base.desc_head[2].weight.detach().cpu().numpy(),
        base.desc_head[2].bias.detach().cpu().numpy(),
    ]

    with open(weights_blob_path, 'wb') as handle:
        handle.write(b'EPH1')
        handle.write(struct.pack('<I', len(tensors)))
        for tensor in tensors:
            _write_tensor_blob(handle, tensor)


class EdgePoint2Encoder(nn.Module):
    """
    ONNX-exportable wrapper for EdgePoint2 E64 — Metis M2 AIPU compatible.

    All internal channel counts are multiples of 64, satisfying:
      - pass_pword_pad: conv IC/OC hardware word alignment
      - tile_dwpu:      AveragePool/Resize minimum channel count
    Slice operations are eliminated; spatial decimation uses AveragePool.

    Outputs (NCHW float32):
      x1_feat: [1, 64,  H/2,  W/2 ]
      x2_feat: [1, 64,  H/8,  W/8 ]
      x3_feat: [1, 64,  H/32, W/32]
    """

    def __init__(self, base: nn.Module):
        super().__init__()

        # ── block1: 1→16ch, rebuilt as 1→64ch then 64→64ch ─────────────────
        # Original: BasicLayer(1,16,4,2,1) + BasicLayer(16,16) + ResNetLayer(16,16)
        # The first conv keeps IC=1 (the network input) but expands OC to 64.
        # pass_pword_pad only needs to pad IC from 1→64 (OC already aligned);
        # no intermediate type mismatch occurs.  Avoids the Pad ONNX op whose
        # channel-padding amount would need to be a multiple of 64 (63 is not).
        self.block1 = nn.Sequential(
            _pad_basic_layer(base.block1[0], 1, 64),   # Conv2d(1,16,4,2,1) → (1,64)
            _pad_basic_layer(base.block1[1], 64, 64),
            _PaddedResNetLayer(base.block1[2], 64, 64),
        )

        # ── block2: 16→48ch, rebuilt as 64→64ch ─────────────────────────────
        self.block2 = _PaddedResNetLayer(base.block2, 64, 64)

        # ── block3: 48→64ch, rebuilt as 64→64ch ─────────────────────────────
        self.block3 = _PaddedResNetLayer(base.block3, 64, 64)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [1, 1, H, W]  float32 in [0, 1]  (grayscale)

        Returns:
            x1_feat: [1, 64,  H/2,  W/2 ]
            x2_feat: [1, 64,  H/8,  W/8 ]
            x3_feat: [1, 64,  H/32, W/32]
        """
        x1  = self.block1(x)                             # [1, 64, H/2,  W/2 ]

        # AveragePool on 64-ch tensors — DWPU-safe (64 % 64 == 0).
        # No strided spatial slices (compiler only supports single-axis stride-1).
        _x2 = F.avg_pool2d(x1,  2, 2)                  # [1, 64, H/4,  W/4 ]
        x2  = self.block2(F.avg_pool2d(_x2, 2, 2))     # [1, 64, H/8,  W/8 ]

        x3  = F.avg_pool2d(x2, 4, 4)                   # [1, 64, H/32, W/32]
        x3  = self.block3(x3)                           # [1, 64, H/32, W/32]

        return x1, x2, x3


def export(edgepoint_repo: str, weights_path: str,
           output_path: str, height: int, width: int):
    print(f"Loading EdgePoint2 E64 from {weights_path} ...")
    base  = load_model(edgepoint_repo, weights_path)
    model = EdgePoint2Encoder(base).eval()

    dummy = torch.randn(1, 1, height, width)
    with torch.no_grad():
        x1_feat, x2_feat, x3_feat = model(dummy)
    print("Sanity check:")
    print(f"  x1_feat shape : {tuple(x1_feat.shape)}")
    print(f"  x2_feat shape : {tuple(x2_feat.shape)}")
    print(f"  x3_feat shape : {tuple(x3_feat.shape)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX → {output_path} ...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=17,
        input_names=['image'],
        output_names=['x1_feat', 'x2_feat', 'x3_feat'],
        dynamic_axes=None,
        do_constant_folding=True,
        verbose=False,
    )
    _materialize_resize_constants_as_initializers(output_path)
    weights_blob_path = str(Path(output_path).with_name('edgepoint_heads.bin'))
    _export_head_weights(base, weights_blob_path)
    print("Export complete.")
    print(f"Exported head weights → {weights_blob_path}")

    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        inp  = np.random.randn(1, 1, height, width).astype(np.float32)
        out  = sess.run(None, {'image': inp})
        print("ONNX Runtime verification:")
        print(f"  x1_feat shape : {out[0].shape}")
        print(f"  x2_feat shape : {out[1].shape}")
        print(f"  x3_feat shape : {out[2].shape}")
        print("ONNX model verified successfully.")
    except ImportError:
        print("onnxruntime not installed — skipping verification.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export EdgePoint2 E64 to ONNX')
    parser.add_argument('--weights',
                        default='weights/E64.pth',
                        help='Path to E64.pth')
    parser.add_argument('--output',
                        default='weights/edgepoint_encoder.onnx',
                        help='Output ONNX path')
    parser.add_argument('--edgepoint-repo',
                        default='/home/Vydar/EdgePoint2',
                        help='Path to the cloned EdgePoint2 repository')
    parser.add_argument('--height', type=int, default=512,
                        help='Input height (must be divisible by 32)')
    parser.add_argument('--width',  type=int, default=512,
                        help='Input width  (must be divisible by 32)')
    args = parser.parse_args()

    assert args.height % 32 == 0, 'height must be divisible by 32 (backbone stride)'
    assert args.width  % 32 == 0, 'width  must be divisible by 32 (backbone stride)'

    export(args.edgepoint_repo, args.weights, args.output, args.height, args.width)
