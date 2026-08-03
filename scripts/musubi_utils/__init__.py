"""
Musubi Utilities Package

This package contains modules for building commands and running different models
through the musubi-trainer framework.

Modules:
    ltx2_cache: Command builders for LTX-Video-2 caching operations
    ltx2_run: Command builders for LTX-Video-2 training operations
    (Future: wan22_cache, wan22_run, etc.)
"""

from .ltx2_cache import LTX2Cache
from .ltx2_run import LTX2Run
from .wan22_cache import WAN22Cache
from .mmh3_cache import MMH3Cache
from .mmh3_run import MMH3Run

# Import utility functions from original musubi_utils
try:
    from .utils import find_last_state_directory
except ImportError:
    # Fallback if utils module doesn't exist yet
    def find_last_state_directory(output_dir, output_name):
        """Fallback implementation for finding last state directory."""
        import os
        if os.path.exists(output_dir):
            state_dirs = []
            for entry in os.listdir(output_dir):
                entry_path = os.path.join(output_dir, entry)
                if os.path.isdir(entry_path) and entry.endswith("-state"):
                    state_dirs.append((os.path.getmtime(entry_path), entry_path))
            if state_dirs:
                state_dirs.sort(key=lambda x: x[0], reverse=True)
                return state_dirs[0][1]
        return None

__all__ = ['LTX2Cache', 'LTX2Run', 'WAN22Cache', 'MMH3Cache', 'MMH3Run', 'find_last_state_directory']

# Model type registry - maps model identifiers to their handler classes
MODEL_HANDLERS = {
    'ltx-video-2': ('LTX2Cache', 'LTX2Run'),
    'ltx2': ('LTX2Cache', 'LTX2Run'),
    'wan22': ('WAN22Cache', 'LTX2Run'),  # Run handler not implemented yet
    'wan2.2': ('WAN22Cache', 'LTX2Run'),
    'minimaxh3': ('MMH3Cache', 'MMH3Run'),  # Cache handler now wired; run handler is build+print only
    'minimax-h3': ('MMH3Cache', 'MMH3Run'),
    'mmh3': ('MMH3Cache', 'MMH3Run'),
}


def get_cache_handler(model_type: str):
    """Get the cache handler class for a given model type."""
    handlers = MODEL_HANDLERS.get(model_type.lower())
    if not handlers:
        raise ValueError(f"No cache handler found for model type: {model_type}")
    module_name, class_name = handlers
    module = __import__(f'musubi_utils.{module_name.lower()}', fromlist=[class_name])
    return getattr(module, class_name)


def get_run_handler(model_type: str):
    """Get the run handler class for a given model type."""
    handlers = MODEL_HANDLERS.get(model_type.lower())
    if not handlers:
        raise ValueError(f"No run handler found for model type: {model_type}")
    module_name, class_name = handlers
    module = __import__(f'musubi_utils.{module_name.lower()}', fromlist=[class_name])
    return getattr(module, class_name)
