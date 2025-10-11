import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from src.utils import load_data, download_images, calculate_smape
from src.feature_extraction import TextFeatureExtractor, ImageFeatureExtractor
from src.models import PricePredictor
import pickle
import os

def train_model(train_path, model_type='ensemble', use_images=False, save_dir='models'):
    """
    Train price prediction model.
    
    Args:
        train_path: Path to training CSV
        model_type: Type of model to train
        use_images: Whether to use image features
        save_dir: Directory to save trained models
    
    Returns:
        tuple: (model, text_extractor, image_extractor, metrics)
    """
    print("="*50)
    print("PRICE PREDICTION MODEL TRAINING")
    print("="*50)
    
    train_df, _ = load_data(train_path=train_path)
    
    if train_df is None:
        raise ValueError(f"Could not load training data from {train_path}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    print(f"  Mean price: ${train_df['price'].mean():.2f}")
    print(f"  Median price: ${train_df['price'].median():.2f}")
    
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)
    
    print("\n1. Extracting text features...")
    text_extractor = TextFeatureExtractor(max_features=3000)
    text_features = text_extractor.fit_transform(train_df['catalog_content'].values)
    print(f"   Text features shape: {text_features.shape}")
    
    image_features = None
    image_extractor = None
    
    if use_images:
        print("\n2. Downloading and extracting image features...")
        image_paths = download_images(train_df, save_dir='images/train')
        print(f"   Downloaded {len(image_paths)} images")
        
        image_extractor = ImageFeatureExtractor()
        image_features = image_extractor.extract_features(image_paths, train_df['sample_id'].values)
        print(f"   Image features shape: {image_features.shape}")
        
        X = np.hstack([text_features, image_features])
    else:
        X = text_features
    
    y = train_df['price'].values
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    
    print("\n" + "="*50)
    print("MODEL TRAINING WITH CROSS-VALIDATION")
    print("="*50)
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        fold_model = PricePredictor(model_type=model_type)
        fold_model.fit(X_train_fold, y_train_fold)
        
        y_pred_val = fold_model.predict(X_val_fold)
        fold_smape = calculate_smape(y_val_fold, y_pred_val)
        
        cv_scores.append(fold_smape)
        print(f"  Fold {fold} SMAPE: {fold_smape:.4f}%")
    
    print(f"\nCross-validation results:")
    print(f"  Mean SMAPE: {np.mean(cv_scores):.4f}%")
    print(f"  Std SMAPE: {np.std(cv_scores):.4f}%")
    
    print("\n" + "="*50)
    print("FINAL MODEL TRAINING")
    print("="*50)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    final_model = PricePredictor(model_type=model_type)
    final_model.fit(X_train, y_train)
    
    y_pred_train = final_model.predict(X_train)
    y_pred_val = final_model.predict(X_val)
    
    train_smape = calculate_smape(y_train, y_pred_train)
    val_smape = calculate_smape(y_val, y_pred_val)
    
    print(f"\nFinal Model Performance:")
    print(f"  Training SMAPE: {train_smape:.4f}%")
    print(f"  Validation SMAPE: {val_smape:.4f}%")
    
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    final_model.save(f'{save_dir}/price_model.pkl')
    
    with open(f'{save_dir}/text_extractor.pkl', 'wb') as f:
        pickle.dump(text_extractor, f)
    print(f"Text extractor saved to {save_dir}/text_extractor.pkl")
    
    if image_extractor:
        with open(f'{save_dir}/image_extractor.pkl', 'wb') as f:
            pickle.dump(image_extractor, f)
        print(f"Image extractor saved to {save_dir}/image_extractor.pkl")
    
    metrics = {
        'cv_mean_smape': np.mean(cv_scores),
        'cv_std_smape': np.std(cv_scores),
        'train_smape': train_smape,
        'val_smape': val_smape,
        'cv_scores': cv_scores
    }
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    return final_model, text_extractor, image_extractor, metrics


if __name__ == "__main__":
    train_model(
        train_path='dataset/train.csv',
        model_type='ensemble',
        use_images=False
    )
