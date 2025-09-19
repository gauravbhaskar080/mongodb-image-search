// MongoDB Image Search - Frontend JavaScript

class ImageSearchApp {
    constructor() {
        this.apiBase = '';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadStats();
        console.log('MongoDB Image Search App initialized');
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleImageUpload();
        });

        // File input change
        document.getElementById('imageFile').addEventListener('change', (e) => {
            this.previewImage(e.target.files[0]);
        });

        // Similar image click events (delegated)
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('similar-image')) {
                this.showImageModal(e.target.src, e.target.alt);
            }
        });

        // Modal close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('image-modal')) {
                e.target.style.display = 'none';
            }
        });

        // Keyboard events
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const modal = document.querySelector('.image-modal');
                if (modal) modal.style.display = 'none';
            }
        });
    }

    async loadStats() {
        try {
            const response = await fetch('/stats');
            if (response.ok) {
                const stats = await response.json();
                this.displayStats(stats);
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
            document.getElementById('stats').innerHTML = `
                <div class="stat-item">
                    <i class="fas fa-exclamation-triangle text-warning"></i>
                    <span class="ms-2">Stats unavailable</span>
                </div>
            `;
        }
    }

    displayStats(stats) {
        const statsContainer = document.getElementById('stats');
        statsContainer.innerHTML = `
            <div class="stat-item">
                <i class="fas fa-images text-primary"></i>
                <span class="ms-2">${stats.total_images} Images</span>
            </div>
            <div class="stat-item">
                <i class="fas fa-database text-info"></i>
                <span class="ms-2">MongoDB Atlas</span>
            </div>
            <div class="stat-item">
                <i class="fas fa-microchip text-success"></i>
                <span class="ms-2">${stats.embedding_dimension}D Embeddings</span>
            </div>
        `;
    }

    previewImage(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please select a valid image file (JPG, PNG, BMP)');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size must be less than 10MB');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const queryImg = document.getElementById('queryImage');
            if (queryImg) {
                queryImg.src = e.target.result;
                queryImg.style.display = 'block';
                document.getElementById('queryImageInfo').textContent = 
                    `${file.name} (${this.formatFileSize(file.size)})`;
            }
        };
        reader.readAsDataURL(file);
    }

    async handleImageUpload() {
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('imageFile');
        const topK = document.getElementById('topK').value;
        const searchBtn = document.getElementById('searchBtn');

        if (!fileInput.files[0]) {
            this.showError('Please select an image file');
            return;
        }

        // Show loading state
        this.showLoading();
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Searching...';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch(`/search?k=${topK}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result, fileInput.files[0]);
                this.showSuccess(`Found ${result.results.length} similar images!`);
            } else {
                throw new Error(result.detail || 'Search failed');
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showError(`Search failed: ${error.message}`);
        } finally {
            this.hideLoading();
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search Similar Images';
        }
    }

    displayResults(searchResult, queryFile) {
        const resultsSection = document.getElementById('resultsSection');
        const similarImagesContainer = document.getElementById('similarImages');
        const resultsInfo = document.getElementById('resultsInfo');

        // Update query image info
        document.getElementById('queryImageInfo').textContent = 
            `${queryFile.name} (${this.formatFileSize(queryFile.size)})`;

        // Update results info
        resultsInfo.innerHTML = `
            <i class="fas fa-clock me-2"></i>
            Search completed in ${searchResult.search_time.toFixed(3)}s • 
            Found ${searchResult.results.length} similar images
        `;

        // Clear previous results
        similarImagesContainer.innerHTML = '';

        // Display similar images
        searchResult.results.forEach((result, index) => {
            const imageCard = this.createImageCard(result, index);
            similarImagesContainer.appendChild(imageCard);
        });

        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    createImageCard(result, index) {
        const col = document.createElement('div');
        col.className = 'col-lg-4 col-md-6';

        const similarity = (result.similarity_score * 100).toFixed(1);
        const scoreClass = this.getSimilarityScoreClass(result.similarity_score);

        col.innerHTML = `
            <div class="card similar-image-card h-100">
                <div class="position-relative">
                    <img src="data:image/jpeg;base64,${result.image_data}" 
                         class="similar-image" 
                         alt="Similar Image ${index + 1}"
                         loading="lazy">
                    <span class="similarity-score ${scoreClass}">
                        ${similarity}%
                    </span>
                </div>
                <div class="card-body">
                    <h6 class="card-title">
                        <i class="fas fa-image me-2"></i>
                        ${result.filename}
                    </h6>
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">
                            <i class="fas fa-fingerprint me-1"></i>
                            ID: ${result.id.substring(0, 8)}...
                        </small>
                        <span class="badge bg-primary">
                            Rank #${index + 1}
                        </span>
                    </div>
                    <div class="mt-2">
                        <div class="progress" style="height: 4px;">
                            <div class="progress-bar ${this.getProgressBarClass(result.similarity_score)}" 
                                 role="progressbar" 
                                 style="width: ${similarity}%"
                                 aria-valuenow="${similarity}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                            </div>
                        </div>
                        <small class="text-muted mt-1 d-block">
                            Similarity: ${similarity}%
                        </small>
                    </div>
                </div>
            </div>
        `;

        return col;
    }

    getSimilarityScoreClass(score) {
        if (score >= 0.8) return 'high';
        if (score >= 0.6) return 'medium';
        return 'low';
    }

    getProgressBarClass(score) {
        if (score >= 0.8) return 'bg-success';
        if (score >= 0.6) return 'bg-warning';
        return 'bg-danger';
    }

    showImageModal(src, alt) {
        let modal = document.querySelector('.image-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.className = 'image-modal';
            modal.innerHTML = '<img src="" alt="">';
            document.body.appendChild(modal);
        }

        const img = modal.querySelector('img');
        img.src = src;
        img.alt = alt;
        modal.style.display = 'flex';
    }

    showLoading() {
        document.getElementById('loadingSection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
    }

    showError(message) {
        this.showAlert(message, 'danger');
    }

    showSuccess(message) {
        this.showAlert(message, 'success');
    }

    showAlert(message, type) {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alert.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 1060;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;
        
        const icon = type === 'success' ? 'check-circle' : 'exclamation-triangle';
        
        alert.innerHTML = `
            <i class="fas fa-${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alert);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageSearchApp();
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// Service worker registration (optional - for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}