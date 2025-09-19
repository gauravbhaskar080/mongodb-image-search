"""
Utility functions for image file discovery.
"""
import os
from typing import List
from PIL import Image

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass

# Supported image extensions
_VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: str) -> List[str]:
    """
    Recursively find all supported image files in a directory.
    
    Args:
        root: Root directory to search
        
    Returns:
        List of absolute paths to image files
    """
    root = os.path.abspath(root)
    imgs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in _VALID_EXT:
                imgs.append(os.path.join(dirpath, fn))
    imgs.sort()
    return imgs
