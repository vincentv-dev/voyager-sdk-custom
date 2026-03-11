# SuperPoint custom ONNX model for Axelera AI Voyager SDK
#
# The ONNX model outputs two tensors:
#   scores:      [1, H, W]          raw score map (post softmax + pixel-shuffle)
#   descriptors: [1, 256, H/8, W/8] L2-normalised dense descriptor map
#
# All keypoint post-processing is handled in superpoint_decoder.py.

import onnx
from pathlib import Path

from ax_models import base_onnx
from axelera import types
from axelera.app import logging_utils, utils

LOG = logging_utils.getLogger(__name__)


class SuperPointONNXModel(base_onnx.AxONNXModel):
    """AxONNXModel subclass for SuperPoint.

    Inherits weight downloading / loading from AxONNXModel.
    No extra logic needed; the ONNX graph already contains the full
    backbone + head computation.
    """

    def init_model_deploy(self, model_info: types.ModelInfo, dataset_config: dict, **kwargs):
        weights = Path(model_info.weight_path)
        if not weights.exists() or (
            model_info.weight_md5 and not utils.md5_validates(weights, model_info.weight_md5)
        ):
            if not model_info.weight_url:
                raise ValueError(
                    f"No suitable weights found for {model_info.name} at {weights} "
                    f"and no weight_url specified"
                )
            try:
                utils.download(model_info.weight_url, weights, model_info.weight_md5)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {weights} from {model_info.weight_url}\n\t{e}"
                ) from None
        LOG.debug(f"Load SuperPoint ONNX model from {weights}")
        self.onnx_model = onnx.load(str(weights))
