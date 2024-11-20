import sys

from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("gt-guidance")
class GTGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        ckpt_path: str = ""
        config_path: str = ""

    cfg: Config

    def configure(self) -> None:
        pass

    def forward(
        self,
        pred: Float[Tensor, "B H W C"],
        prompt_utils,
        imgs: Float[Tensor, "B H W C"],
        **kwargs
    ):
        loss = torch.nn.functional.mse_loss(pred, imgs, reduction="mean")
        return {
            "loss_sds": loss,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass
