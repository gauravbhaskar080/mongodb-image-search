#!/usr/bin/env python3
"""
MongoDB-based Image Similarity Search Script
Load images and embeddings from MongoDB Atlas and perform similarity search.
"""

import os
import sys
import glob
import numpy as np
from typing import List, Tuple, Dict, Any
from pymongo import MongoClient
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import ImageEmbedder
from src.search import search_topk
from config import (MONGO_CONNECTION, DATABASE_NAME, COLLECTION_NAME, 
                   QUERY_DIR, OUTPUT_DIR, DEFAULT_TOP_K, setup_directories)

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass


class MongoImageSearcher:
    """MongoDB-based image similarity search."""
    
    def __init__(self, 
                 connection_string: str,
                 database_name: str = "image_search_db",
                 collection_name: str = "images"):
        """
        Initialize MongoDB connection and load embeddings.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database
            collection_name: Name of the collection for images
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
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
                    '_id': doc['_id'],
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
            print(f"Embedding shape: {self.embeddings.shape}")
        else:
            raise ValueError("No embeddings found in MongoDB collection!")
    
    def embed_query_image(self, query_path: str) -> np.ndarray:
        """Generate embedding for a query image."""
        try:
            with Image.open(query_path) as img:
                img_rgb = img.convert('RGB')
                embedding = self.embedder.embed_pil(img_rgb)
                return embedding
        except Exception as e:
            print(f"Error processing query image {query_path}: {e}")
            return None
    
    def search_similar_images(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search for similar images using cosine similarity.
        
        Args:
            query_embedding: Query image embedding
            k: Number of top results to return
            
        Returns:
            indices: List of indices of similar images
            similarities: List of similarity scores
            metadata: List of metadata for similar images
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded!")
        
        # Use the existing search_topk function
        indices, similarities = search_topk(query_embedding, self.embeddings, k=k)
        
        # Get metadata for the results
        top_metadata = [self.image_metadata[i] for i in indices]
        
        return indices.tolist(), similarities.tolist(), top_metadata
    
    def visualize_search_results(self, query_path: str, similar_metadata: List[Dict], 
                               similarities: List[float], save_path: str = None):
        """
        Visualize query image and search results.
        
        Args:
            query_path: Path to query image
            similar_metadata: List of metadata for similar images
            similarities: List of similarity scores
            save_path: Optional path to save the visualization
        """
        n_results = len(similar_metadata)
        fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 4))
        
        if n_results == 0:
            axes = [axes]
        
        # Show query image
        try:
            query_img = Image.open(query_path)
            axes[0].imshow(query_img)
            axes[0].set_title(f"Query: {os.path.basename(query_path)}", fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            # Add red border to query image
            rect = patches.Rectangle((0, 0), query_img.width-1, query_img.height-1, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error loading\nquery image\n{str(e)}", 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title("Query Image (Error)", fontsize=10)
        
        # Show similar images
        for i, (metadata, similarity) in enumerate(zip(similar_metadata, similarities)):
            try:
                if os.path.exists(metadata['filepath']):
                    similar_img = Image.open(metadata['filepath'])
                    axes[i + 1].imshow(similar_img)
                else:
                    # Image file not found locally, show placeholder
                    axes[i + 1].text(0.5, 0.5, f"Image not found\nlocally", 
                                    ha='center', va='center', transform=axes[i + 1].transAxes)
                
                axes[i + 1].set_title(f"#{i+1}: {metadata['filename']}\nSimilarity: {similarity:.4f}", 
                                    fontsize=9)
                axes[i + 1].axis('off')
                
                # Add green border to results
                if os.path.exists(metadata['filepath']):
                    img_for_border = Image.open(metadata['filepath'])
                    rect = patches.Rectangle((0, 0), img_for_border.width-1, img_for_border.height-1, 
                                           linewidth=2, edgecolor='green', facecolor='none')
                    axes[i + 1].add_patch(rect)
                
            except Exception as e:
                axes[i + 1].text(0.5, 0.5, f"Error loading\n{metadata['filename']}\n{str(e)}", 
                                ha='center', va='center', transform=axes[i + 1].transAxes)
                axes[i + 1].set_title(f"#{i+1}: {metadata['filename']} (Error)", fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def find_query_images(query_dir: str = "query-image") -> List[str]:
    """Find all query images in the specified directory."""
    if not os.path.exists(query_dir):
        print(f"Query directory '{query_dir}' not found!")
        return []
    
    query_patterns = [
        f"{query_dir}/*.jpg",
        f"{query_dir}/*.jpeg", 
        f"{query_dir}/*.png",
        f"{query_dir}/*.bmp"
    ]
    
    query_images = []
    for pattern in query_patterns:
        query_images.extend(glob.glob(pattern))
    
    return sorted(query_images)


def main():
    """Main function to run MongoDB-based image search."""
    print("=" * 70)
    print("MongoDB-based Image Similarity Search")
    print("=" * 70)
    
    # Setup directories
    setup_directories()
    
    # Find query images
    query_images = find_query_images(QUERY_DIR)
    if not query_images:
        print("No query images found! Please add images to the 'query-image' directory.")
        return
    
    print(f"Found {len(query_images)} query images:")
    for img in query_images:
        print(f"  - {os.path.basename(img)}")
    
    # Initialize MongoDB searcher
    try:
        print(f"\nConnecting to MongoDB Atlas...")
        searcher = MongoImageSearcher(MONGO_CONNECTION, DATABASE_NAME, COLLECTION_NAME)
        
        # Process each query image
        for query_path in query_images:
            print(f"\n{'='*70}")
            print(f"Processing: {os.path.basename(query_path)}")
            print(f"{'='*70}")
            
            # Generate query embedding
            print("Generating query embedding...")
            query_embedding = searcher.embed_query_image(query_path)
            
            if query_embedding is None:
                print("Failed to generate embedding for query image")
                continue
            
            # Search for similar images
            print(f"Searching for top {DEFAULT_TOP_K} similar images...")
            indices, similarities, metadata = searcher.search_similar_images(
                query_embedding, k=DEFAULT_TOP_K
            )
            
            # Display results
            print(f"\nTop {DEFAULT_TOP_K} similar images:")
            print("-" * 50)
            for i, (idx, sim, meta) in enumerate(zip(indices, similarities, metadata)):
                print(f"{i+1}. {meta['filename']}")
                print(f"   Similarity: {sim:.4f}")
                print(f"   Dimensions: {meta['width']}x{meta['height']}")
                print(f"   File size: {meta['file_size']:,} bytes")
                print(f"   Path: {meta['filepath']}")
                print()
            
            # Create visualization
            query_name = os.path.splitext(os.path.basename(query_path))[0]
            viz_path = os.path.join(OUTPUT_DIR, f"mongodb_search_{query_name}.png")
            
            print(f"Creating visualization...")
            searcher.visualize_search_results(
                query_path, metadata, similarities, save_path=viz_path
            )
        
        searcher.close()
        print(f"\n✓ MongoDB search completed successfully!")
        print(f"✓ Visualizations saved in '{OUTPUT_DIR}' directory")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()