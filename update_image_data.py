#!/usr/bin/env python3
"""
Script to update existing MongoDB documents with base64 image data.
This is needed for the web application to display search results.
"""

import os
import sys
import io
import base64
from PIL import Image
from pymongo import MongoClient

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import MONGO_CONNECTION, DATABASE_NAME, COLLECTION_NAME

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass

def update_images_with_data():
    """Update existing MongoDB documents with base64 image data."""
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_CONNECTION)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    print("Finding documents without image_data...")
    
    # Find all documents that don't have image_data field
    cursor = collection.find({"image_data": {"$exists": False}})
    documents = list(cursor)
    
    print(f"Found {len(documents)} documents to update")
    
    updated_count = 0
    for doc in documents:
        try:
            filepath = doc.get('filepath')
            if not filepath or not os.path.exists(filepath):
                print(f"  ✗ Skipping {doc.get('filename', 'unknown')} - file not found")
                continue
            
            print(f"  Processing {doc.get('filename', 'unknown')}...")
            
            # Load and encode image
            with Image.open(filepath) as img:
                # Convert to RGB and save as JPEG
                img_rgb = img.convert('RGB')
                buffered = io.BytesIO()
                img_rgb.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Update the document
            result = collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"image_data": img_base64}}
            )
            
            if result.modified_count > 0:
                print(f"  ✓ Updated {doc.get('filename', 'unknown')}")
                updated_count += 1
            else:
                print(f"  ✗ Failed to update {doc.get('filename', 'unknown')}")
                
        except Exception as e:
            print(f"  ✗ Error processing {doc.get('filename', 'unknown')}: {e}")
            continue
    
    print(f"\n✓ Successfully updated {updated_count} documents with image data")
    
    # Verify the update
    total_with_images = collection.count_documents({"image_data": {"$exists": True}})
    print(f"✓ Total documents with image data: {total_with_images}")
    
    client.close()

if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB Image Data Update Script")
    print("=" * 60)
    
    try:
        update_images_with_data()
        print("\n✓ Update completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)