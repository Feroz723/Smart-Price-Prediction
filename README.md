# Amazon ML Challenge 2025 - Smart Product Pricing Solution

A machine learning solution for predicting e-commerce product prices using text and image features with ensemble methods, optimized for SMAPE metric.

## 🎯 Challenge Overview

**Objective**: Predict product prices based on catalog content and product images

**Dataset**: 
- Training: 75K products with prices
- Test: 75K products for prediction

**Evaluation**: SMAPE (Symmetric Mean Absolute Percentage Error) - lower is better

## 🚀 Quick Start

### 1. Setup
All dependencies are automatically installed. The project includes:
- Python 3.11
- scikit-learn, XGBoost (always available)
- LightGBM, CatBoost (optional - may require system libraries)
- pandas, numpy, nltk, pillow

**Note**: If LightGBM or CatBoost fail to load due to missing system libraries (libgomp.so.1), the system will automatically fall back to XGBoost and Random Forest models. The ensemble will still work effectively with the available models.

### 2. Run Demo
```bash
python demo.py
```

### 3. Train Model (with your data)
```bash
# Text features only (faster)
python main.py --mode train --train-path dataset/train.csv

# With image features (better accuracy, slower)
python main.py --mode train --train-path dataset/train.csv --use-images
```

### 4. Generate Predictions
```bash
python main.py --mode predict --test-path dataset/test.csv
```

### 5. Complete Pipeline
```bash
# Train and predict in one command
python main.py --mode both
```

## 📁 Project Structure

```
├── src/
│   ├── utils.py              # Helper functions (I/O, SMAPE, image download)
│   ├── feature_extraction.py # Text (TF-IDF, IPQ) & image features
│   ├── models.py             # ML models and ensemble
│   ├── train.py              # Training with cross-validation
│   └── predict.py            # Prediction generation
├── dataset/                  # Place train.csv and test.csv here
├── images/                   # Downloaded product images
├── models/                   # Saved trained models
├── output/                   # test_out.csv predictions
├── main.py                   # Main entry point
├── demo.py                   # Demo with sample data
└── Documentation.md          # Detailed methodology
```

## 🔬 Methodology

### Feature Engineering
1. **Text Features**
   - TF-IDF vectorization (3000-5000 features)
   - Item Pack Quantity (IPQ) extraction (value, unit)
   - Text statistics (length, word count, keywords)

2. **Image Features** (optional)
   - Color statistics (RGB mean/std)
   - Image dimensions
   - Basic visual patterns

### Models
- **XGBoost**: Gradient boosting (30% weight)
- **LightGBM**: Fast gradient boosting (30% weight)
- **CatBoost**: Categorical boosting (25% weight)
- **Random Forest**: Tree ensemble (15% weight)

### Training Strategy
- 5-fold cross-validation
- 80/20 train-validation split
- SMAPE optimization
- Hyperparameter tuning

## 📊 Usage Examples

### Basic Training
```bash
python main.py --mode train
```

### Advanced Training
```bash
python main.py \
  --mode train \
  --train-path dataset/train.csv \
  --model-type ensemble \
  --use-images
```

### Prediction Only
```bash
python main.py \
  --mode predict \
  --test-path dataset/test.csv \
  --output-path output/test_out.csv
```

### Model Type Options
- `ensemble` (default): Combines all models
- `xgboost`: XGBoost only
- `lightgbm`: LightGBM only
- `catboost`: CatBoost only
- `rf`: Random Forest only

## 📈 Expected Performance

- **Cross-validation SMAPE**: Varies by data
- **Validation SMAPE**: Reported after training
- **Prediction Range**: $0.01 - $XXX.XX
- **Output Format**: CSV with columns: sample_id, price

## 🔧 How It Works

1. **Data Loading**: Reads CSV files with product information
2. **Feature Extraction**: 
   - Extracts TF-IDF features from catalog text
   - Parses Item Pack Quantity (value and unit)
   - Optionally downloads and processes images
3. **Model Training**: 
   - Trains multiple models with cross-validation
   - Creates weighted ensemble
   - Saves models to disk
4. **Prediction**: 
   - Loads trained models
   - Extracts features from test data
   - Generates price predictions
   - Saves to test_out.csv

## 📝 Output Format

The prediction output (`test_out.csv`) contains:
```csv
sample_id,price
217392,45.67
209156,23.45
...
```

## 🛠️ Customization

### Adjust Model Hyperparameters
Edit `src/models.py` to modify:
- Number of estimators
- Learning rate
- Max depth
- Ensemble weights

### Change Feature Count
Edit `src/feature_extraction.py`:
```python
TextFeatureExtractor(max_features=5000)  # Increase/decrease
```

### Add New Features
Extend the `TextFeatureExtractor` or `ImageFeatureExtractor` classes

## 📋 Requirements Checklist

- ✅ MIT/Apache 2.0 licensed models (< 8B parameters)
- ✅ No external LLM APIs used
- ✅ No external price lookup
- ✅ SMAPE evaluation metric
- ✅ Proper output format
- ✅ Well-commented source code
- ✅ Documentation included

## 🐛 Troubleshooting

### Issue: Image download fails
**Solution**: Use `--use-images` flag with caution. Text-only model works well.

### Issue: Out of memory
**Solution**: Reduce `max_features` in TextFeatureExtractor or use a single model instead of ensemble.

### Issue: Missing train.csv or test.csv
**Solution**: Place your data files in the `dataset/` directory with exact names `train.csv` and `test.csv`.

## 📚 Documentation

See `Documentation.md` for:
- Detailed methodology
- Experiments conducted
- Model architecture
- Results and conclusions
- Future improvements

## 🏆 Submission

1. Train model: `python main.py --mode train --train-path dataset/train.csv`
2. Generate predictions: `python main.py --mode predict --test-path dataset/test.csv`
3. Submit `output/test_out.csv` to challenge portal
4. Include this code and `Documentation.md` with submission

## 📄 License

This project uses only MIT/Apache 2.0 licensed libraries and complies with all Amazon ML Challenge 2025 rules.

---

**Good luck with the challenge! 🚀**
