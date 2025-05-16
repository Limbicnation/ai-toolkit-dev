# Models package initialization
# This file enables proper importing of model components

# Prevent circular imports 
from typing import TYPE_CHECKING, Any, Dict, Union, Optional

# Import commonly used base types
if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion