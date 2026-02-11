"""ComfyUI nodes for SoulX-Singer."""

from .model_loader import SoulXSingerModelLoader
from .simple_synthesizer import SoulXSingerSimple
from .advanced_synthesizer import SoulXSingerAdvanced

__all__ = ['SoulXSingerModelLoader', 'SoulXSingerSimple', 'SoulXSingerAdvanced']
