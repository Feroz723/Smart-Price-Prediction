import numpy as np
import pandas as pd
from src.utils import load_data, download_images, save_predictions
from src.feature_extraction import TextFeatureExtractor, ImageFeatureExtractor
from src.models import PricePredictor
import pickle
import os

def predict_prices(test_path, model_dir='models', use_images=False, output_path='output/test_out.csv'):
    """
    Generate price predictions for test data.
    
    Args:
        test_path: Path to test CSV
        model_dir: Directory containing trained models
        use_images: Whether to use image features
        output_path: Path to save predictions
    """
    print("="*50)
    print("PRICE PREDICTION")
    print("="*50)
    
    _, test_df = load_data(test_path=test_path)
    
    if test_df is None:
        raise ValueError(f"Could not load test data from {test_path}")
    
    print(f"\nTest set size: {len(test_df)}")
    
    print("\n" + "="*50)
    print("LOADING TRAINED MODELS")
    print("="*50)
    
    model = PricePredictor.load(f'{model_dir}/price_model.pkl')
    
    with open(f'{model_dir}/text_extractor.pkl', 'rb') as f:
        text_extractor = pickle.load(f)
    print("Text extractor loaded")
    
    image_extractor = None
    if use_images and os.path.exists(f'{model_dir}/image_extractor.pkl'):
        with open(f'{model_dir}/image_extractor.pkl', 'rb') as f:
            image_extractor = pickle.load(f)
        print("Image extractor loaded")
    
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)
    
    print("\n1. Extracting text features...")
    text_features = text_extractor.transform(test_df['catalog_content'].values)
    print(f"   Text features shape: {text_features.shape}")
    
    if use_images and image_extractor:
        print("\n2. Downloading and extracting image features...")
        image_paths = download_images(test_df, save_dir='images/test')
        print(f"   Downloaded {len(image_paths)} images")
        
        image_features = image_extractor.extract_features(image_paths, test_df['sample_id'].values)
        print(f"   Image features shape: {image_features.shape}")
        
        X_test = np.hstack([text_features, image_features])
    else:
        X_test = text_features
    
    print(f"\nFinal feature matrix shape: {X_test.shape}")
    
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    predictions = model.predict(X_test)
    
    predictions = np.maximum(predictions, 0.01)
    
    print(f"\nPrediction Statistics:")
    print(f"  Min predicted price: ${predictions.min():.2f}")
    print(f"  Max predicted price: ${predictions.max():.2f}")
    print(f"  Mean predicted price: ${predictions.mean():.2f}")
    print(f"  Median predicted price: ${np.median(predictions):.2f}")
    
    print("\n" + "="*50)
    print("SAVING PREDICTIONS")
    print("="*50)
    
    save_predictions(test_df['sample_id'].values, predictions, output_path)
    
    output_df = pd.read_csv(output_path)
    print(f"\nOutput file verification:")
    print(f"  Columns: {list(output_df.columns)}")
    print(f"  Shape: {output_df.shape}")
    print(f"\nFirst 5 predictions:")
    print(output_df.head())
    
    print("\n" + "="*50)
    print("PREDICTION COMPLETE!")
    print("="*50)
    
    return predictions


if __name__ == "__main__":
    predict_prices(
        test_path='dataset/test.csv',
        model_dir='models',
        use_images=False,
        output_path='output/test_out.csv'
    )
