#!/usr/bin/env python3
"""
Demo script to show the ML pipeline with sample data
"""

import os
import pandas as pd
import numpy as np
from src.utils import calculate_smape

def create_sample_data():
    """Create sample training data from sample_test.csv"""
    
    if not os.path.exists('dataset/sample_test.csv'):
        print("Error: sample_test.csv not found")
        return
    
    sample_df = pd.read_csv('dataset/sample_test.csv')
    
    np.random.seed(42)
    sample_df['price'] = np.random.uniform(5, 100, len(sample_df))
    
    sample_df.to_csv('dataset/sample_train.csv', index=False)
    print(f"Created sample training data with {len(sample_df)} samples")
    
    return sample_df

def demo():
    """Run a simple demonstration of the ML pipeline"""
    
    print("="*70)
    print(" "*15 + "AMAZON ML CHALLENGE 2025 - DEMO")
    print(" "*12 + "Smart Product Pricing Solution")
    print("="*70)
    
    print("\n1. Creating sample training data...")
    sample_df = create_sample_data()
    
    if sample_df is None:
        return
    
    print("\n2. Data Statistics:")
    print(f"   Samples: {len(sample_df)}")
    print(f"   Features: catalog_content, image_link")
    print(f"   Target: price")
    
    print("\n3. Sample Data Preview:")
    print(sample_df[['sample_id', 'price']].head(10))
    
    print("\n4. Text Feature Examples:")
    for idx, row in sample_df.head(3).iterrows():
        text = row['catalog_content']
        if pd.notna(text) and len(text) > 0:
            print(f"\n   Sample {row['sample_id']}:")
            print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
    
    print("\n5. SMAPE Metric Demonstration:")
    y_true = np.array([100, 50, 75, 120])
    y_pred = np.array([110, 48, 80, 115])
    
    smape = calculate_smape(y_true, y_pred)
    print(f"   True prices: {y_true}")
    print(f"   Predicted prices: {y_pred}")
    print(f"   SMAPE: {smape:.2f}%")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    
    print("\nTo train the actual model:")
    print("  python main.py --mode train --train-path dataset/train.csv")
    
    print("\nTo generate predictions:")
    print("  python main.py --mode predict --test-path dataset/test.csv")
    
    print("\nTo do both:")
    print("  python main.py --mode both")
    print()

if __name__ == "__main__":
    demo()
