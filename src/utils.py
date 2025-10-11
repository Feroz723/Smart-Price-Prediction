import os
import time
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
from tqdm import tqdm

def download_images(df, image_column='image_link', save_dir='images', max_retries=3):
    """
    Download images from URLs with retry mechanism to handle throttling.
    
    Args:
        df: DataFrame containing image URLs
        image_column: Column name containing image URLs
        save_dir: Directory to save downloaded images
        max_retries: Maximum number of retry attempts
    
    Returns:
        dict: Mapping of sample_id to local image path
    """
    os.makedirs(save_dir, exist_ok=True)
    image_paths = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        sample_id = row['sample_id']
        image_url = row[image_column]
        
        if pd.isna(image_url) or not image_url:
            continue
            
        image_path = os.path.join(save_dir, f"{sample_id}.jpg")
        
        if os.path.exists(image_path):
            image_paths[sample_id] = image_path
            continue
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img.save(image_path, 'JPEG')
                
                image_paths[sample_id] = image_path
                time.sleep(0.1)
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed to download image for {sample_id}: {str(e)}")
    
    return image_paths


def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Formula: SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        float: SMAPE score (0-200%, lower is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    smape = np.mean(numerator / (denominator + 1e-10)) * 100
    
    return smape


def load_data(train_path=None, test_path=None):
    """
    Load training and test datasets.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
    
    Returns:
        tuple: (train_df, test_df)
    """
    train_df = None
    test_df = None
    
    if train_path and os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        print(f"Loaded {len(train_df)} training samples")
    
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Loaded {len(test_df)} test samples")
    
    return train_df, test_df


def save_predictions(sample_ids, predictions, output_path='output/test_out.csv'):
    """
    Save predictions in the required format.
    
    Args:
        sample_ids: Array of sample IDs
        predictions: Array of predicted prices
        output_path: Path to save output CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(output_df)}")
