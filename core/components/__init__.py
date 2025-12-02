"""
Components package for Enhanced Ghibli Processor
"""
# Import non-model-dependent components first
from .prompt_engineer import PromptEngineer
from .preprocessing_pipeline import PreprocessingPipeline
from .postprocessing_pipeline import PostprocessingPipeline

__all__ = [
    'PromptEngineer',
    'PreprocessingPipeline',
    'PostprocessingPipeline',
    'ModelManager',
    'GenerationEngine'
]

# Lazy import for model-dependent components
def __getattr__(name):
    if name == 'ModelManager':
        from .model_manager import ModelManager
        return ModelManager
    elif name == 'GenerationEngine':
        from .generation_engine import GenerationEngine
        return GenerationEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
