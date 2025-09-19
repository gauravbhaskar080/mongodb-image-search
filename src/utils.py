"""
Utility functions for the image embedding search system.
"""
import os
from PIL import Image

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)
