#!/usr/bin/env python3
"""
Amazon ML Challenge 2025 - Smart Product Pricing
Main script for training and prediction
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from src.train import train_model
from src.predict import predict_prices

def create_sample_training_data():
    """Create sample training data from sample_test.csv if needed."""
    
    sample_train_path = 'dataset/sample_train.csv'
    
    if os.path.exists(sample_train_path):
        return sample_train_path
    
    sample_test_path = 'dataset/sample_test.csv'
    if not os.path.exists(sample_test_path):
        return None
    
    print("Creating sample training data from sample_test.csv...")
    sample_df = pd.read_csv(sample_test_path)
    
    np.random.seed(42)
    sample_df['price'] = np.random.uniform(5, 100, len(sample_df))
    
    sample_df.to_csv(sample_train_path, index=False)
    print(f"Sample training data created with {len(sample_df)} samples")
    
    return sample_train_path


def main():
    """Main entry point for the application."""
    
    parser = argparse.ArgumentParser(
        description='Amazon ML Challenge 2025 - Price Prediction Model'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'both'],
        default='both',
        help='Mode: train, predict, or both (default: both)'
    )
    
    parser.add_argument(
        '--train-path',
        type=str,
        default='dataset/train.csv',
        help='Path to training data CSV'
    )
    
    parser.add_argument(
        '--test-path',
        type=str,
        default='dataset/test.csv',
        help='Path to test data CSV'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['ensemble', 'xgboost', 'lightgbm', 'catboost', 'rf'],
        default='ensemble',
        help='Model type to use (default: ensemble)'
    )
    
    parser.add_argument(
        '--use-images',
        action='store_true',
        help='Use image features (requires downloading images)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='output/test_out.csv',
        help='Path to save predictions'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" "*15 + "AMAZON ML CHALLENGE 2025")
    print(" "*12 + "Smart Product Pricing Solution")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Model type: {args.model_type}")
    print(f"  Use images: {args.use_images}")
    print(f"  Train path: {args.train_path}")
    print(f"  Test path: {args.test_path}")
    print(f"  Output path: {args.output_path}")
    print("="*70 + "\n")
    
    if args.mode in ['train', 'both']:
        if not os.path.exists(args.train_path):
            print(f"\nWarning: Training data not found at {args.train_path}")
            print("Please ensure the training data (train.csv) is available for actual training.")
            
            print("\nCreating sample training data for demonstration...")
            sample_path = create_sample_training_data()
            
            if sample_path is None:
                print(f"Error: Cannot create sample data. Please provide train.csv")
                sys.exit(1)
            
            args.train_path = sample_path
            print(f"Using sample training data: {args.train_path}")
        
        print("\n" + "="*70)
        print("TRAINING PHASE")
        print("="*70 + "\n")
        
        model, text_extractor, image_extractor, metrics = train_model(
            train_path=args.train_path,
            model_type=args.model_type,
            use_images=args.use_images,
            save_dir='models'
        )
        
        print("\nTraining metrics saved:")
        print(f"  Cross-validation mean SMAPE: {metrics['cv_mean_smape']:.4f}%")
        print(f"  Validation SMAPE: {metrics['val_smape']:.4f}%")
    
    if args.mode in ['predict', 'both']:
        if not os.path.exists('models/price_model.pkl'):
            print("\nError: Trained model not found. Please train the model first.")
            sys.exit(1)
        
        if not os.path.exists(args.test_path):
            print(f"\nWarning: Test data not found at {args.test_path}")
            print("Using sample test data for demonstration...")
            args.test_path = 'dataset/sample_test.csv'
        
        print("\n" + "="*70)
        print("PREDICTION PHASE")
        print("="*70 + "\n")
        
        predictions = predict_prices(
            test_path=args.test_path,
            model_dir='models',
            use_images=args.use_images,
            output_path=args.output_path
        )
    
    print("\n" + "="*70)
    print(" "*25 + "PROCESS COMPLETE!")
    print("="*70 + "\n")
    
    print("Next steps:")
    if args.mode in ['train', 'both']:
        print("  ✓ Model trained and saved to models/")
    if args.mode in ['predict', 'both']:
        print(f"  ✓ Predictions saved to {args.output_path}")
        print(f"  → Submit {args.output_path} to the challenge portal")
    print("\n")


if __name__ == "__main__":
    main()
