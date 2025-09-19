# MongoDB-Based Image Embedding Search

A clean and efficient image similarity search system using MongoDB Atlas for cloud storage and ResNet50 embeddings.

## 🚀 Features

- **Cloud-Based Storage**: Images and embeddings stored in MongoDB Atlas
- **Deep Learning Embeddings**: ResNet50-based 2048-dimensional feature vectors
- **Real-Time Search**: Fast cosine similarity search with visualization
- **Batch Processing**: Efficient handling of multiple images
- **Duplicate Prevention**: SHA256 hash-based deduplication
- **Rich Metadata**: File dimensions, sizes, timestamps, and similarity scores

## 📁 Project Structure

```
image-embedding-search/
├── src/                          # Core modules
│   ├── embeddings.py            # ResNet50 image embedding generation
│   ├── search.py                # Similarity search functions
│   ├── index.py                 # Image file discovery utilities
│   ├── config.py                # Device and model configuration
│   └── utils.py                 # General utilities
├── data/
│   └── images/                  # Source images for indexing
├── query-image/                 # Query images for testing
├── outputs/
│   └── queries/                 # Search result visualizations
├── config.py                    # MongoDB and system configuration
├── store_to_mongodb.py          # Store images and embeddings to MongoDB
├── mongodb_search.py            # Search similar images from MongoDB
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **MongoDB Configuration**:
   - Update `config.py` with your MongoDB Atlas connection string
   - The default configuration uses a cloud database

3. **Add Images**:
   - Place images to index in `data/images/`
   - Place query images in `query-image/`
   - Supported formats: JPG, JPEG, PNG, BMP

## 📊 Usage

### 1. Store Images to MongoDB

```bash
python store_to_mongodb.py
```

This will:
- Scan all images in `data/images/`
- Generate ResNet50 embeddings for each image
- Store images metadata and embeddings in MongoDB Atlas
- Skip duplicates automatically

### 2. Search Similar Images

```bash
python mongodb_search.py
```

This will:
- Load embeddings from MongoDB Atlas
- Process all query images in `query-image/`
- Find the top-3 most similar images
- Create visual comparisons in `outputs/queries/`
- Display similarity scores and metadata

## 🔧 Configuration

Edit `config.py` to customize:

```python
# MongoDB settings
MONGO_CONNECTION = "your_mongodb_connection_string"
DATABASE_NAME = "image_search_db"
COLLECTION_NAME = "images"

# Processing settings
DEFAULT_BATCH_SIZE = 8
DEFAULT_TOP_K = 3

# Directory settings
IMAGES_DIR = "data/images"
QUERY_DIR = "query-image"
OUTPUT_DIR = "outputs/queries"
```

## 📈 Performance

- **GPU Acceleration**: Automatically uses CUDA if available
- **Batch Processing**: Configurable batch size for memory efficiency
- **Indexed Search**: MongoDB indexes on hash and filename for fast queries
- **Normalized Embeddings**: L2-normalized 2048-dimensional vectors

## 🎯 Example Results

For a query image `briyani.jpg`, the system might return:
1. `000074.jpg` - Similarity: 0.6492
2. `000073.jpg` - Similarity: 0.6328  
3. `000043.jpg` - Similarity: 0.5666

Visual comparisons are automatically saved as PNG files.

## 🔍 System Architecture

```
Images → ResNet50 Embeddings → MongoDB Atlas → Similarity Search → Visualization
```

1. **Embedding Generation**: ResNet50 extracts 2048-dimensional features
2. **Cloud Storage**: MongoDB Atlas stores metadata + embeddings
3. **Search Engine**: Cosine similarity ranking of stored embeddings
4. **Visualization**: Side-by-side query and result comparisons

## 📝 MongoDB Document Structure

```json
{
  "_id": "ObjectId",
  "filename": "image.jpg",
  "filepath": "/path/to/image.jpg", 
  "width": 640,
  "height": 480,
  "file_size": 45678,
  "image_hash": "sha256_hash",
  "embedding": [0.123, -0.456, ...],  // 2048 dimensions
  "embedding_dimension": 2048,
  "processed_time": "2025-01-01T12:00:00Z"
}
```

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure MongoDB connection in `config.py`
4. Add images to `data/images/`
5. Run `python store_to_mongodb.py` to index images
6. Add query images to `query-image/`
7. Run `python mongodb_search.py` to search

The system is ready for production use and scales to thousands of images efficiently!