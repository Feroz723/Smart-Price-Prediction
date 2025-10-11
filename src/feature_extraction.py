import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class TextFeatureExtractor:
    """Extract features from catalog content text."""
    
    def __init__(self, max_features=5000):
        """
        Initialize text feature extractor.
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.fitted = False
    
    def extract_ipq(self, text):
        """
        Extract Item Pack Quantity (IPQ) from catalog content.
        
        Args:
            text: Catalog content text
        
        Returns:
            dict: Extracted numerical features
        """
        features = {
            'value': None,
            'unit': None,
            'count': None,
            'ounce': None,
            'pound': None,
            'gram': None,
            'pack_size': None
        }
        
        if pd.isna(text):
            return features
        
        text_lower = text.lower()
        
        value_pattern = r'value:\s*(\d+\.?\d*)'
        value_match = re.search(value_pattern, text_lower)
        if value_match:
            features['value'] = float(value_match.group(1))
        
        unit_patterns = {
            'count': r'(\d+\.?\d*)\s*count',
            'ounce': r'(\d+\.?\d*)\s*(oz|ounce)',
            'pound': r'(\d+\.?\d*)\s*(lb|pound)',
            'gram': r'(\d+\.?\d*)\s*(g|gram)',
            'pack_size': r'pack\s*of\s*(\d+)'
        }
        
        for key, pattern in unit_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                features[key] = float(match.group(1))
        
        return features
    
    def extract_text_stats(self, text):
        """
        Extract statistical features from text.
        
        Args:
            text: Catalog content text
        
        Returns:
            dict: Text statistics
        """
        if pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'bullet_points': 0,
                'has_brand': 0,
                'has_organic': 0,
                'has_natural': 0,
                'avg_word_length': 0
            }
        
        words = text.split()
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'bullet_points': text.count('Bullet Point'),
            'has_brand': 1 if any(brand in text.lower() for brand in ['brand', 'premium', 'gourmet']) else 0,
            'has_organic': 1 if 'organic' in text.lower() else 0,
            'has_natural': 1 if 'natural' in text.lower() else 0,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0
        }
    
    def fit_transform(self, texts):
        """
        Fit TF-IDF vectorizer and transform texts.
        
        Args:
            texts: List of text documents
        
        Returns:
            np.array: TF-IDF features combined with numerical features
        """
        texts_clean = [str(t) if not pd.isna(t) else '' for t in texts]
        
        tfidf_features = self.tfidf.fit_transform(texts_clean).toarray()
        self.fitted = True
        
        ipq_features = []
        text_stats = []
        
        for text in texts:
            ipq = self.extract_ipq(text)
            stats = self.extract_text_stats(text)
            
            ipq_values = [v if v is not None else 0 for v in ipq.values()]
            stats_values = list(stats.values())
            
            ipq_features.append(ipq_values)
            text_stats.append(stats_values)
        
        ipq_features = np.array(ipq_features)
        text_stats = np.array(text_stats)
        
        combined_features = np.hstack([tfidf_features, ipq_features, text_stats])
        
        return combined_features
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of text documents
        
        Returns:
            np.array: TF-IDF features combined with numerical features
        """
        if not self.fitted:
            raise ValueError("Extractor not fitted. Call fit_transform first.")
        
        texts_clean = [str(t) if not pd.isna(t) else '' for t in texts]
        
        tfidf_features = self.tfidf.transform(texts_clean).toarray()
        
        ipq_features = []
        text_stats = []
        
        for text in texts:
            ipq = self.extract_ipq(text)
            stats = self.extract_text_stats(text)
            
            ipq_values = [v if v is not None else 0 for v in ipq.values()]
            stats_values = list(stats.values())
            
            ipq_features.append(ipq_values)
            text_stats.append(stats_values)
        
        ipq_features = np.array(ipq_features)
        text_stats = np.array(text_stats)
        
        combined_features = np.hstack([tfidf_features, ipq_features, text_stats])
        
        return combined_features


class ImageFeatureExtractor:
    """Extract basic features from product images."""
    
    def __init__(self):
        """Initialize image feature extractor."""
        self.target_size = (224, 224)
    
    def extract_basic_features(self, image_path):
        """
        Extract basic statistical features from image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            np.array: Image features
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img)
            
            features = [
                np.mean(img_array[:, :, 0]),
                np.std(img_array[:, :, 0]),
                np.mean(img_array[:, :, 1]),
                np.std(img_array[:, :, 1]),
                np.mean(img_array[:, :, 2]),
                np.std(img_array[:, :, 2]),
                np.mean(img_array),
                np.std(img_array),
                img_array.shape[0],
                img_array.shape[1]
            ]
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return np.zeros(10)
    
    def extract_features(self, image_paths_dict, sample_ids):
        """
        Extract features from multiple images.
        
        Args:
            image_paths_dict: Dictionary mapping sample_id to image path
            sample_ids: List of sample IDs to process
        
        Returns:
            np.array: Image features for all samples
        """
        features_list = []
        
        for sample_id in sample_ids:
            if sample_id in image_paths_dict:
                features = self.extract_basic_features(image_paths_dict[sample_id])
            else:
                features = np.zeros(10)
            
            features_list.append(features)
        
        return np.array(features_list)
