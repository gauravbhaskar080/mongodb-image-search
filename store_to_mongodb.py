#!/usr/bin/env python3
"""
Script to store image data and embeddings from data/images folder into MongoDB.
"""

import os
import sys
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from pymongo import MongoClient
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import ImageEmbedder
from src.index import list_images
from config import MONGO_CONNECTION, DATABASE_NAME, COLLECTION_NAME, IMAGES_DIR, DEFAULT_BATCH_SIZE

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass

class ImageMongoDBStorage:
    """Handles storing image data and embeddings into MongoDB."""
    
    def __init__(self, 
                 connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "image_search_db",
                 collection_name: str = "images"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database
            collection_name: Name of the collection for images
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Create index on image_hash for faster lookups
        self.collection.create_index("image_hash")
        self.collection.create_index("filename")
        
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate SHA256 hash of the image file."""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from image file and encode image data."""
        try:
            # Get file stats
            stat = os.stat(image_path)
            
            # Get image dimensions and encode image data
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                # Convert image to base64 for storage
                import io
                import base64
                
                # Convert to RGB if needed and save as JPEG
                img_rgb = img.convert('RGB')
                buffered = io.BytesIO()
                img_rgb.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            metadata = {
                'filename': os.path.basename(image_path),
                'filepath': image_path,
                'file_size': stat.st_size,
                'width': width,
                'height': height,
                'mode': mode,
                'format': format_name,
                'created_time': datetime.fromtimestamp(stat.st_ctime),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'image_hash': self._calculate_image_hash(image_path),
                'processed_time': datetime.now(),
                'image_data': img_base64  # Store base64 encoded image data
            }
            
            return metadata
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return None
    
    def store_images_batch(self, image_paths: List[str], batch_size: int = 8) -> int:
        """
        Store multiple images and their embeddings in MongoDB.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for embedding generation
            
        Returns:
            Number of successfully stored images
        """
        print(f"Building embedding model...")
        embedder = ImageEmbedder()
        
        stored_count = 0
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}: {len(batch_paths)} images")
            
            try:
                # Generate embeddings for the batch
                embeddings = embedder.embed_paths(batch_paths, batch_size=len(batch_paths))
                
                # Process each image in the batch
                for j, image_path in enumerate(batch_paths):
                    try:
                        # Get image metadata
                        metadata = self._get_image_metadata(image_path)
                        if metadata is None:
                            continue
                        
                        # Check if image already exists (by hash)
                        existing = self.collection.find_one({"image_hash": metadata['image_hash']})
                        if existing:
                            print(f"  Skipping {metadata['filename']} (already exists)")
                            continue
                        
                        # Prepare document for MongoDB
                        document = {
                            **metadata,
                            'embedding': embeddings[j].tolist(),  # Convert numpy array to list
                            'embedding_dimension': len(embeddings[j])
                        }
                        
                        # Insert into MongoDB
                        result = self.collection.insert_one(document)
                        if result.inserted_id:
                            print(f"  ✓ Stored {metadata['filename']} (ID: {result.inserted_id})")
                            stored_count += 1
                        else:
                            print(f"  ✗ Failed to store {metadata['filename']}")
                            
                    except Exception as e:
                        print(f"  ✗ Error processing {image_path}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        return stored_count
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the image collection."""
        total_count = self.collection.count_documents({})
        
        # Get sample document for schema info
        sample = self.collection.find_one({})
        embedding_dim = sample.get('embedding_dimension', 0) if sample else 0
        
        # Calculate storage size (approximate)
        stats = self.db.command("collStats", self.collection.name)
        
        return {
            'total_images': total_count,
            'embedding_dimension': embedding_dim,
            'collection_size_bytes': stats.get('size', 0),
            'avg_document_size_bytes': stats.get('avgObjSize', 0),
            'storage_size_bytes': stats.get('storageSize', 0)
        }
    
    def list_stored_images(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List stored images with basic info."""
        cursor = self.collection.find(
            {}, 
            {'filename': 1, 'filepath': 1, 'file_size': 1, 'width': 1, 'height': 1, 'processed_time': 1}
        ).limit(limit)
        
        return list(cursor)


def main():
    """Main function to run the image storage process."""
    print("=" * 60)
    print("Image Embedding MongoDB Storage Script")
    print("=" * 60)
    
    # Check if images directory exists
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory '{IMAGES_DIR}' not found!")
        return
    
    # Find all image files using existing function
    print(f"Scanning for images in '{IMAGES_DIR}'...")
    image_paths = list_images(IMAGES_DIR)
    
    if not image_paths:
        print("No image files found!")
        return
    
    print(f"Found {len(image_paths)} image files:")
    for path in image_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Initialize MongoDB storage
    try:
        print(f"\nConnecting to MongoDB...")
        storage = ImageMongoDBStorage(MONGO_CONNECTION, DATABASE_NAME, COLLECTION_NAME)
        print(f"Connected to database '{DATABASE_NAME}', collection '{COLLECTION_NAME}'")
        
        # Store images
        print(f"\nStarting image processing (batch size: {DEFAULT_BATCH_SIZE})...")
        stored_count = storage.store_images_batch(image_paths, batch_size=DEFAULT_BATCH_SIZE)
        
        # Show results
        print(f"\n" + "=" * 60)
        print(f"RESULTS")
        print(f"=" * 60)
        print(f"Successfully stored: {stored_count} images")
        
        # Show collection statistics
        stats = storage.get_collection_stats()
        print(f"\nCollection Statistics:")
        print(f"  Total images in DB: {stats['total_images']}")
        print(f"  Embedding dimension: {stats['embedding_dimension']}")
        print(f"  Collection size: {stats['collection_size_bytes']:,} bytes")
        print(f"  Storage size: {stats['storage_size_bytes']:,} bytes")
        
        # Show some stored images
        print(f"\nRecently stored images:")
        stored_images = storage.list_stored_images(limit=5)
        for img in stored_images:
            print(f"  - {img['filename']} ({img['width']}x{img['height']}, {img['file_size']:,} bytes)")
        
        storage.close()
        print(f"\n✓ Process completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()