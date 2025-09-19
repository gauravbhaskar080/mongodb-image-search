"""
FastAPI Web Application for MongoDB Image Search
Provides REST API and web interface for image similarity search.
"""

import os
import sys
import io
import base64
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import ImageEmbedder
from src.search import search_topk
from config import (MONGO_CONNECTION, DATABASE_NAME, COLLECTION_NAME, 
                   DEFAULT_TOP_K, SUPPORTED_FORMATS)

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass


class ImageSearchService:
    """Service for image similarity search using MongoDB."""
    
    def __init__(self):
        self.client = MongoClient(MONGO_CONNECTION)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        # Load embeddings and metadata from MongoDB
        self.embeddings = None
        self.image_metadata = []
        self.embedder = ImageEmbedder()
        
        self._load_embeddings_from_db()
    
    def _load_embeddings_from_db(self):
        """Load all embeddings and metadata from MongoDB."""
        print("Loading embeddings from MongoDB...")
        
        # Get all documents from the collection
        cursor = self.collection.find({})
        embeddings_list = []
        metadata_list = []
        
        for doc in cursor:
            if 'embedding' in doc:
                embeddings_list.append(doc['embedding'])
                # Store relevant metadata
                metadata = {
                    '_id': str(doc['_id']),
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'width': doc.get('width', 0),
                    'height': doc.get('height', 0),
                    'file_size': doc.get('file_size', 0)
                }
                metadata_list.append(metadata)
        
        if embeddings_list:
            self.embeddings = np.array(embeddings_list, dtype=np.float32)
            self.image_metadata = metadata_list
            print(f"Loaded {len(embeddings_list)} embeddings from MongoDB")
        else:
            raise ValueError("No embeddings found in MongoDB collection!")
    
    def embed_query_image(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for a query image."""
        try:
            img_rgb = image.convert('RGB')
            embedding = self.embedder.embed_pil(img_rgb)
            return embedding
        except Exception as e:
            print(f"Error processing query image: {e}")
            return None
    
    def search_similar_images(self, query_embedding: np.ndarray, k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Search for similar images and return results with metadata."""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded!")
        
        # Use the existing search_topk function
        indices, similarities = search_topk(query_embedding, self.embeddings, k=k)
        
        # Get metadata for the results
        results = []
        for i, (idx, similarity) in enumerate(zip(indices, similarities)):
            metadata = self.image_metadata[idx].copy()
            metadata['similarity'] = float(similarity)
            metadata['rank'] = i + 1
            results.append(metadata)
        
        return {
            'total_results': len(results),
            'results': results
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_count = self.collection.count_documents({})
        
        # Get sample document for schema info
        sample = self.collection.find_one({})
        embedding_dim = sample.get('embedding_dimension', 0) if sample else 0
        
        return {
            'total_images': total_count,
            'embedding_dimension': embedding_dim,
            'status': 'ready' if self.embeddings is not None else 'not_loaded'
        }


# Initialize FastAPI app
app = FastAPI(
    title="MongoDB Image Search API",
    description="AI-powered image similarity search using MongoDB and ResNet50 embeddings",
    version="1.0.0"
)

# Initialize search service
search_service = ImageSearchService()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with image upload interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stats")
async def get_stats():
    """Get database and service statistics."""
    return search_service.get_stats()


@app.post("/search")
async def search_images(
    file: UploadFile = File(...),
    k: int = 3
):
    """
    Search for similar images by uploading a query image.
    
    - **file**: Image file to search for similar images
    - **k**: Number of similar images to return (default: 3)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported image format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Read and process the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Generate embedding
        query_embedding = search_service.embed_query_image(image)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate image embedding")
        
        # Search for similar images
        search_results = search_service.search_similar_images(query_embedding, k=k)
        
        # Add image data to results
        results_with_images = []
        for result in search_results['results']:
            # Load image from filepath and convert to base64
            try:
                # Get the document from MongoDB to get the original image data
                from bson import ObjectId
                doc = search_service.collection.find_one({"_id": ObjectId(result['_id'])})
                if doc and 'image_data' in doc:
                    result['image_data'] = doc['image_data']
                else:
                    # If no stored image data, create a placeholder
                    result['image_data'] = ""
                
                result['id'] = result['_id']
                result['similarity_score'] = result['similarity']
                results_with_images.append(result)
            except Exception as e:
                print(f"Error loading image for result {result['_id']}: {e}")
                continue
        
        search_time = time.time() - start_time
        
        return {
            "results": results_with_images,
            "total_results": len(results_with_images),
            "search_time": search_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    """Get image metadata by ID."""
    try:
        from bson import ObjectId
        result = search_service.collection.find_one({"_id": ObjectId(image_id)})
        if not result:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Convert ObjectId to string for JSON response
        result['_id'] = str(result['_id'])
        # Remove embedding for lighter response
        result.pop('embedding', None)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "mongodb-image-search",
        "embeddings_loaded": search_service.embeddings is not None,
        "total_images": len(search_service.image_metadata) if search_service.image_metadata else 0
    }


if __name__ == "__main__":
    import uvicorn
    
    # Create required directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Get port from environment variable (for deployment) or use 8000 (for local)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("Starting MongoDB Image Search API...")
    print(f"Total images in database: {len(search_service.image_metadata)}")
    print(f"Server will be available at: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port)