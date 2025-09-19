"""
Common configuration for the MongoDB image search system.
"""
import os
from PIL import Image

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass

# MongoDB Configuration
MONGO_CONNECTION = os.getenv(
    "MONGO_CONNECTION", 
    "mongodb+srv://gaurav:bhaskar@cluster0.abwlx6b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "image_search_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "images")

# Processing Configuration  
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "8"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))

# Directory Configuration
IMAGES_DIR = os.getenv("IMAGES_DIR", "data/images")
QUERY_DIR = os.getenv("QUERY_DIR", "query-image")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/queries")

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}


def setup_directories():
    """Ensure required directories exist."""
    for dir_path in [OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)