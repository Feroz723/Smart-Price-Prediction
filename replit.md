# Amazon ML Challenge 2025 - Smart Product Pricing Solution

## Overview

This is a machine learning solution for predicting e-commerce product prices based on catalog text content and product images. The system uses an ensemble of gradient boosting models (XGBoost, LightGBM, CatBoost) combined with Random Forest to predict prices, optimized for the SMAPE (Symmetric Mean Absolute Percentage Error) metric.

**Core Purpose**: Predict product prices from text descriptions and images without external price lookups, meeting the constraint of using only MIT/Apache 2.0 licensed models with ≤8B parameters.

**Dataset**: 75K training products with prices, 75K test products for prediction.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### 1. Application Structure

**Entry Points**:
- `main.py` - Primary CLI interface for training and prediction workflows
- `demo.py` - Demonstration script using sample data
- Three execution modes: train-only, predict-only, or both (complete pipeline)

**Modular Design**:
- `src/feature_extraction.py` - Text and image feature engineering
- `src/models.py` - Ensemble model architecture and individual regressors
- `src/train.py` - Model training with cross-validation
- `src/predict.py` - Inference pipeline
- `src/utils.py` - Shared utilities (data loading, image downloading, metrics)

**Design Rationale**: Separation of concerns allows independent testing of feature extraction, model training, and prediction. The modular structure makes it easy to swap feature extractors or models without affecting other components.

### 2. Feature Engineering Architecture

**Text Feature Extraction** (`TextFeatureExtractor`):
- **TF-IDF Vectorization**: Extracts 3000-5000 most important terms using bigrams (1-2 word combinations)
- **Item Pack Quantity (IPQ) Parsing**: Regex-based extraction of numerical values and units (ounces, pounds, grams, count, fl oz) from product descriptions
- **Text Statistics**: Length, word count, and other derived metrics

**Image Feature Extraction** (`ImageFeatureExtractor`):
- Downloads images from URLs with retry mechanism (handles throttling)
- Extracts visual features from product images
- Optional feature - system works with text-only features

**Design Decision**: Text features are primary because they're more reliable and faster to process. Image features are optional add-ons that improve accuracy but increase processing time. This allows users to choose speed vs. accuracy tradeoff.

### 3. Model Architecture

**Ensemble Strategy**:
- **Primary Models**: XGBoost, LightGBM, CatBoost (gradient boosting variants)
- **Secondary Model**: Random Forest (diversity in ensemble)
- **Fallback Mechanism**: If LightGBM/CatBoost fail due to missing system libraries (libgomp.so.1), automatically falls back to XGBoost + Random Forest

**Model Configuration**:
- All models use 200 estimators with depth 8
- Learning rate: 0.05 for gradient boosters
- Subsample/feature sampling: 0.8 (prevent overfitting)
- Custom SMAPE-optimized predictions via weighted averaging

**Why Ensemble?**: Different algorithms capture different patterns. XGBoost excels at structured data, LightGBM is memory-efficient, CatBoost handles categorical features well, and Random Forest provides robustness. Weighted ensemble reduces variance and improves generalization.

**Alternatives Considered**:
- Single model (XGBoost only): Simpler but less robust
- Deep learning: Requires more data and compute, violates 8B parameter constraint
- Linear models: Too simple for complex pricing patterns

### 4. Training Pipeline

**Data Flow**:
1. Load training CSV (75K products)
2. Extract text features (TF-IDF + IPQ parsing)
3. Optionally extract image features
4. Combine features into single matrix
5. Train ensemble models with cross-validation
6. Save models and feature extractors as pickle files

**Cross-Validation**: K-Fold validation used during training to estimate model performance and prevent overfitting

**Model Persistence**: All components (models, feature extractors) saved to `models/` directory for reproducible inference

### 5. Prediction Pipeline

**Data Flow**:
1. Load test CSV (75K products)
2. Load trained models and feature extractors
3. Extract features using same transformers as training
4. Generate predictions from ensemble
5. Post-process (ensure positive prices)
6. Save to CSV matching required format

**Output Format**: CSV with `sample_id` and `price` columns, exactly matching test set sample IDs

### 6. Data Management

**Storage Structure**:
- `dataset/` - CSV files (train.csv, test.csv, sample files)
- `models/` - Serialized models and feature extractors
- `images/` - Downloaded product images (if using image features)
- `output/` - Prediction results

**Data Loading**: Centralized in `utils.py` with validation and error handling

## External Dependencies

### Python Libraries

**Core ML Stack**:
- `scikit-learn` - Feature extraction (TF-IDF), baseline models (Random Forest, Ridge), train/test splitting
- `xgboost` - Primary gradient boosting model (always available)
- `lightgbm` - Memory-efficient gradient boosting (optional, graceful degradation)
- `catboost` - Category-optimized gradient boosting (optional, graceful degradation)

**Data Processing**:
- `pandas` - CSV loading, data manipulation
- `numpy` - Numerical operations, array handling
- `nltk` - Text preprocessing (stop words)

**Image Processing**:
- `Pillow (PIL)` - Image loading, format conversion, resizing
- `requests` - HTTP client for downloading images from URLs

**Utilities**:
- `pickle` - Model serialization
- `tqdm` - Progress bars for long operations

**Dependency Philosophy**: The system has a graceful degradation strategy. Core functionality (text-based prediction) works with just scikit-learn, XGBoost, pandas, and numpy. LightGBM and CatBoost enhance accuracy but aren't required. Image features are optional for better predictions.

### External APIs/Services

**Image URLs**: Product images are hosted on Amazon's CDN (e.g., `https://m.media-amazon.com/images/...`)
- Download mechanism includes retry logic to handle throttling
- Images cached locally to avoid repeated downloads
- System works without images (text-only mode)

### System Requirements

- Python 3.11
- System libraries: May require `libgomp.so.1` for LightGBM/CatBoost (auto-fallback if missing)
- No GPU required (CPU-only training and inference)
- No database dependencies (all data in CSV format)

### Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**: Built-in calculation in `utils.py`
- Lower is better
- Symmetric handling of over/under predictions
- Range: 0-200% (typically 0-100% for good models)