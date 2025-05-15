import os
import copy
import inspect
import json
import math
import random
import re
import time
import warnings
from datetime import datetime
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnx
import safetensors
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from torchvision import transforms
from tqdm import tqdm

import toolkit.util.extract as extract
from toolkit.basic import force_list, list_files
from toolkit.clip_manager import get_clip_model_list, make_clip_config
from toolkit.clip_model import make_clip_model, set_clip_embedding_tokens
from toolkit.config_modules import (ClipConfig, ImageProcessorConfig, TextEncoderConfig, VAEConfig,
                                UNetConfig, StageConfig, VertexAIConfig, ParameterConfig,
                                TextProcessorConfig)
from toolkit.custom_compile import compile_model
from toolkit.gemba_utils import scale_tensor, GemBARUPipeline
from toolkit.lora import LoRANetwork, convert_to_kohya
from toolkit.loss.depth_loss import DepthLoss
from toolkit.pbar import tqdm_joblib
from toolkit.sd_device_states_presets import STATES_PRESETS
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
from einops import rearrange, repeat
import torch
from toolkit.pipelines import CustomStableDiffusionXLPipeline, CustomStableDiffusionPipeline, \
    StableDiffusionKDiffusionXLPipeline, StableDiffusionXLRefinerPipeline, FluxWithCFGPipeline, \
    FluxAdvancedControlPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, T2IAdapter, DDPMScheduler, \
    StableDiffusionXLAdapterPipeline, StableDiffusionAdapterPipeline, DiffusionPipeline, PixArtTransformer2DModel, \
    StableDiffusionXLImg2ImgPipeline, LCMScheduler, Transformer2DModel, AutoencoderTiny, ControlNetModel, \
    StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, StableDiffusion3Pipeline, \
    StableDiffusion3Img2ImgPipeline, PixArtSigmaPipeline, AuraFlowPipeline, AuraFlowTransformer2DModel, FluxPipeline, \
    FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, Lumina2Text2ImgPipeline, \
    FluxControlPipeline
from toolkit.models.lumina2 import Lumina2Transformer2DModel
import diffusers
from diffusers import \
    StableDiffusionKDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline, AutoencoderKL, \
    UNet2DConditionModel, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DEISMultistepScheduler, \
    UniPCMultistepScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, \
    AutoPipelineForImage2Image

try:
    from diffusers.models.unets.unet_stable_diffusion import UNetMidBlock2DCrossAttn, CrossAttnDownBlock2D, DownBlock2D, \
        CrossAttnUpBlock2D, UpBlock2D, get_down_block, get_up_block
    from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
except Exception as e:
    warnings.warn('Importing from diffusers UNet parts failed, we are using an older version of diffusers')
    from diffusers.models.unet_2d_condition import UNetMidBlock2DCrossAttn, CrossAttnDownBlock2D, DownBlock2D, \
        CrossAttnUpBlock2D, UpBlock2D, get_down_block, get_up_block
    from diffusers.models.transformer_2d import Transformer2DModelOutput

try:
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5Tokenizer, \
        AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    warnings.warn("Importing CLIPTextModel failed, we are using transformers which doesn't have it")

from transformers import PretrainedConfig
try:
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
except Exception as e:
    warnings.warn('Importing from transformers modeling_outputs failed, we are using an older version of transformers')

try:
    from toolkit.stable_diffusion_reference_only import StableDiffusionReferencePipeline
except Exception as e:
    warnings.warn("Importing StableDiffusionReferencePipeline failed")

try:
    from toolkit.models.sd3 import SD3Pipeline
except Exception as e:
    warnings.warn("Importing SD3Pipeline failed")

try:
    from toolkit.models.SDXL_lightning import SDXLLightningPipeline
except Exception as e:
    warnings.warn("Importing SDXLLightningPipeline failed")

from toolkit.train_tools import auto_detect_unet_dim, expand_features_adapter, get_2d_feature_slices