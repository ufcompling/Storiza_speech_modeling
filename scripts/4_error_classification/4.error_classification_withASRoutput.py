#!/usr/bin/env python3

"""
Joint ASR + Error Detection using Wav2Vec2 Features
Combines acoustic features with ASR-derived linguistic features
for improved error detection in child speech.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2Model, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support
)
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path
from tqdm import tqdm
import warnings
import hashlib
import argparse
warnings.filterwarnings('ignore')

# ===============================================
# DATA LOADING AND PREPROCESSING
# ===============================================

def load_and_prepare_data(data_file):
    """Load and prepare the dataset"""
    print(f"Loading data from {data_file}...")
    full_data = pd.read_csv(data_file)
    
    # Shuffle data
    word_segments_data = full_data.sample(frac=1, random_state=42)
    
    # Extract paths
    audio_path_list = word_segments_data['Path'].tolist()
    
    # Process error categories
    word_segments_data["Error Category"] = word_segments_data["Error Category"].apply(
        lambda x: [category.strip() for category in x.split("+") if category != 'Mixed Error']
    )
    error_category_list = word_segments_data['Error Category'].tolist()
    
    # Process error labels
    word_segments_data['Error Labels'] = word_segments_data['Error Labels'].apply(
        lambda labels: labels.strip('[]').split(', ') if labels != '[]' else ['NONE']
    )
    error_label_list = word_segments_data['Error Labels'].tolist()
    
    # Consolidate error categories
    error_map = {
        'Grammatical': 'Grammatical Error',
        'Orthographic Sub.': 'Orthographic Error',
        'Phonological': 'Phonological Error',
        'Run-on': 'Run-on Word',
        'Structural': 'Structural Error',
        'Visual Tracking': 'Visual Tracking Error',
        'Contraction/Shortening': 'Correct'
    }
    
    modified_error_category_list = []
    for categories in error_category_list:
        categories = [error_map.get(cat, cat) for cat in categories]
        modified_error_category_list.append(categories)
    
    print(f"Loaded {len(audio_path_list)} samples")
    print_error_distribution(modified_error_category_list)
    
    return audio_path_list, modified_error_category_list

def print_error_distribution(error_category_list):
    """Print distribution of error categories"""
    error_category_dist = {}
    for categories in error_category_list:
        for category in categories:
            error_category_dist[category] = error_category_dist.get(category, 0) + 1
    
    sorted_dist = dict(sorted(error_category_dist.items(), key=lambda x: x[1], reverse=True))
    print("\nError Category Distribution:")
    total = sum(sorted_dist.values())
    for category, count in sorted_dist.items():
        print(f"  {category}: {count} ({count/total:.2%})")

def create_binary_labels(target_error_category, error_category_list):
    """Create binary labels for a target error category"""
    binary_labels = []
    for categories in error_category_list:
        if target_error_category in categories:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels

def validate_audio_dataset(audio_paths, labels, min_duration=0.1):
    """Validate audio dataset and identify problematic files"""
    clean_paths = []
    clean_labels = []
    problem_files = []
    
    print(f"Validating {len(audio_paths)} audio files...")
    
    for i, (path, label) in enumerate(zip(audio_paths, labels)):
        try:
            # Check if file exists
            if not os.path.exists(path):
                problem_files.append((path, "File not found"))
                continue
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size == 0:
                problem_files.append((path, "Empty file (0 bytes)"))
                continue
            
            # Try to load audio
            try:
                waveform, sample_rate = torchaudio.load(path)
            except Exception as e:
                problem_files.append((path, f"Cannot load audio: {str(e)}"))
                continue
            
            # Check if audio has content
            if waveform.numel() == 0:
                problem_files.append((path, "No audio data"))
                continue
            
            # Check duration
            duration = waveform.shape[-1] / sample_rate
            if duration < min_duration:
                problem_files.append((path, f"Too short: {duration:.3f}s"))
                continue
            
            # Check for all zeros (silent)
            if torch.all(waveform == 0):
                problem_files.append((path, "Silent audio (all zeros)"))
                continue
            
            # File is good
            clean_paths.append(path)
            clean_labels.append(label)
            
        except Exception as e:
            problem_files.append((path, f"Validation error: {str(e)}"))
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Validated {i + 1}/{len(audio_paths)} files...")
    
    # Print summary
    print(f"\nDataset Validation Summary:")
    print(f"✅ Valid files: {len(clean_paths)}")
    print(f"❌ Problem files: {len(problem_files)}")
    
    if problem_files:
        with open('problem_files.txt', 'w') as f:
            for path, issue in problem_files:
                f.write(f"{path}: {issue}\n")
        print(f"Problem files saved to: problem_files.txt")
    
    return clean_paths, clean_labels, problem_files

# ===============================================
# JOINT ASR + ERROR DETECTION CLASSIFIER
# ===============================================

class JointASRErrorClassifier:
    """Classifier that combines acoustic and ASR-derived linguistic features"""
    
    def __init__(self, 
                 acoustic_model_name="facebook/wav2vec2-large-xlsr-53",
                 asr_model_name="facebook/wav2vec2-base-960h",
                 cache_dir="audio_features_cache"):
        """
        Initialize the joint classifier
        
        Args:
            acoustic_model_name: Pre-trained wav2vec2 model for acoustic features
            asr_model_name: Pre-trained ASR model for linguistic features
            cache_dir: Directory for caching features
        """
        print("Initializing JointASRErrorClassifier...")
        print(f"Acoustic model: {acoustic_model_name}")
        print(f"ASR model: {asr_model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load acoustic feature extractor
        print("Loading acoustic model...")
        self.acoustic_processor = Wav2Vec2FeatureExtractor.from_pretrained(acoustic_model_name)
        self.acoustic_model = Wav2Vec2Model.from_pretrained(acoustic_model_name)
        self.acoustic_model.to(self.device)
        self.acoustic_model.eval()
        
        # Load ASR model
        print("Loading ASR model...")
        self.asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
        self.asr_model.to(self.device)
        self.asr_model.eval()
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize classifiers with imbalance handling
        self.classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=42,
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=200,
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ),
            'svm': SVC(
                kernel='rbf',
                random_state=42,
                class_weight='balanced',
                probability=True,
                gamma='scale'
            )
        }
        
        self.scaler = StandardScaler()
        self.best_classifier = None
        self.optimal_threshold = 0.5
        
    def load_audio(self, audio_path, target_sr=16000, max_length=None):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Truncate if specified
            if max_length is not None:
                max_samples = int(max_length * target_sr)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_acoustic_features(self, audio):
        """Extract acoustic features using wav2vec2"""
        try:
            inputs = self.acoustic_processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.acoustic_model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Global average pooling across time
                features = torch.mean(hidden_states, dim=1).squeeze()
                features = features.cpu().numpy()
            
            return features
        except Exception as e:
            print(f"Error extracting acoustic features: {e}")
            return None
    
    def extract_asr_features(self, audio):
        """Extract ASR-derived linguistic features"""
        try:
            inputs = self.asr_processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                asr_outputs = self.asr_model(**inputs)
                logits = asr_outputs.logits
                
                # Feature 1-3: Confidence statistics
                max_probs = torch.softmax(logits, dim=-1).max(dim=-1).values
                confidence_mean = max_probs.mean().cpu().item()
                confidence_std = max_probs.std().cpu().item()
                confidence_min = max_probs.min().cpu().item()
                
                # Feature 4-5: Entropy (uncertainty measure)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropy_mean = entropy.mean().cpu().item()
                entropy_max = entropy.max().cpu().item()
                
                # Feature 6: Predicted text length
                pred_ids = torch.argmax(logits, dim=-1)
                pred_text = self.asr_processor.batch_decode(pred_ids)[0]
                text_length = len(pred_text.split())
                
                # Feature 7: Blank token ratio (CTC specific)
                blank_token_id = self.asr_processor.tokenizer.pad_token_id
                blank_ratio = (pred_ids == blank_token_id).float().mean().cpu().item()
                
                # Feature 8: Confidence variance over time
                confidence_var = max_probs.var().cpu().item()
                
                asr_features = np.array([
                    confidence_mean,
                    confidence_std,
                    confidence_min,
                    entropy_mean,
                    entropy_max,
                    text_length,
                    blank_ratio,
                    confidence_var
                ])
                
            return asr_features
            
        except Exception as e:
            print(f"Error extracting ASR features: {e}")
            return None
    
    def extract_joint_features(self, audio):
        """Extract both acoustic and ASR features"""
        # Extract acoustic features
        acoustic_features = self.extract_acoustic_features(audio)
        if acoustic_features is None:
            return None
        
        # Extract ASR features
        asr_features = self.extract_asr_features(audio)
        if asr_features is None:
            # Fall back to acoustic only
            return acoustic_features
        
        # Concatenate features
        joint_features = np.concatenate([acoustic_features, asr_features])
        return joint_features
    
    def _get_cache_key(self, audio_path, max_length, use_joint):
        """Generate unique cache key"""
        file_stat = os.stat(audio_path)
        key_string = f"{audio_path}_{file_stat.st_mtime}_{max_length}_{use_joint}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_to_cache(self, cache_key, features):
        """Save features to cache"""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_file, features)
    
    def _load_from_cache(self, cache_key):
        """Load features from cache"""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None
    
    def prepare_dataset(self, audio_path_list, error_category_list, 
                       target_error_category='Correct', max_length=2,
                       use_joint_features=True, features_save_path=None):
        """
        Prepare dataset with feature extraction
        
        Args:
            audio_path_list: List of audio file paths
            error_category_list: List of error categories
            target_error_category: Target category for binary classification
            max_length: Maximum audio length in seconds
            use_joint_features: Whether to use joint ASR+acoustic features
            features_save_path: Path to save/load features
        
        Returns:
            features, labels, filenames
        """
        # Try to load pre-saved features
        if features_save_path and os.path.exists(features_save_path):
            print(f"Loading pre-saved features from {features_save_path}")
            return self.load_features_dataset(features_save_path)
        
        features_list = []
        labels_list = []
        filenames = []
        
        binary_labels = create_binary_labels(target_error_category, error_category_list)
        
        print(f"\nPreparing dataset:")
        print(f"  Target error: {target_error_category}")
        print(f"  Joint features: {use_joint_features}")
        print(f"  Max audio length: {max_length}s")
        print(f"  Total samples: {len(audio_path_list)}")
        
        cache_hits = 0
        cache_misses = 0
        
        for i, audio_path in enumerate(tqdm(audio_path_list)):
            # Check cache
            cache_key = self._get_cache_key(audio_path, max_length, use_joint_features)
            features = self._load_from_cache(cache_key)
            
            if features is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                # Load audio
                audio = self.load_audio(audio_path, max_length=max_length)
                if audio is None:
                    continue
                
                # Extract features
                if use_joint_features:
                    features = self.extract_joint_features(audio)
                else:
                    features = self.extract_acoustic_features(audio)
                
                if features is None:
                    continue
                
                # Cache features
                self._save_to_cache(cache_key, features)
            
            features_list.append(features)
            labels_list.append(binary_labels[i])
            filenames.append(Path(audio_path).name)
        
        print(f"\nCache performance: {cache_hits} hits, {cache_misses} misses")
        print(f"Successfully processed: {len(features_list)}/{len(audio_path_list)} files")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\nDataset summary:")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Correct (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"  Error (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        
        # Save features if path provided
        if features_save_path:
            self.save_features_dataset(X, y, filenames, features_save_path)
        
        return X, y, filenames
    
    def save_features_dataset(self, X, y, filenames, save_path):
        """Save feature dataset"""
        dataset = {
            'features': X,
            'labels': y,
            'filenames': filenames,
            'feature_dim': X.shape[1]
        }
        joblib.dump(dataset, save_path)
        print(f"Features saved to: {save_path}")
    
    def load_features_dataset(self, load_path):
        """Load feature dataset"""
        dataset = joblib.load(load_path)
        print(f"Features loaded from: {load_path}")
        print(f"  Samples: {len(dataset['features'])}")
        print(f"  Feature dimension: {dataset['feature_dim']}")
        return dataset['features'], dataset['labels'], dataset['filenames']
    
    def train(self, X, y, test_size=0.2, cv_folds=5):
        """
        Train classifiers with optimal threshold tuning
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set proportion
            cv_folds: Number of CV folds
        
        Returns:
            Dictionary of results for each classifier
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Calculate class weights for XGBoost
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        self.classifiers['xgboost'].scale_pos_weight = pos_weight
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n{'='*60}")
        print("TRAINING CLASSIFIERS")
        print('='*60)
        print(f"Training set: {len(y_train)} samples")
        print(f"  Correct (0): {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
        print(f"  Error (1): {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        print(f"Test set: {len(y_test)} samples")
        print(f"Class imbalance ratio: {pos_weight:.2f}")
        
        results = {}
        best_f1_error = 0
        best_model_name = None
        
        for name, classifier in self.classifiers.items():
            print(f"\n{'-'*60}")
            print(f"Training: {name}")
            print('-'*60)
            
            # Cross-validation
            cv_scores = cross_val_score(
                classifier, X_train_scaled, y_train, 
                cv=cv_folds, scoring='f1', n_jobs=-1
            )
            print(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            # Train on full training set
            classifier.fit(X_train_scaled, y_train)
            
            # Get probability predictions
            if hasattr(classifier, 'predict_proba'):
                y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = classifier.decision_function(X_test_scaled)
            
            # Find optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Predict with optimal threshold
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, labels=[0, 1]
            )
            
            # Store results
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'optimal_threshold': optimal_threshold,
                'model': classifier,
                'precision_correct': precision[0],
                'recall_correct': recall[0],
                'f1_correct': f1[0],
                'precision_error': precision[1],
                'recall_error': recall[1],
                'f1_error': f1[1],
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"Optimal threshold: {optimal_threshold:.3f}")
            print(f"\nCorrect words (class 0):")
            print(f"  Precision: {precision[0]:.3f}")
            print(f"  Recall:    {recall[0]:.3f}")
            print(f"  F1:        {f1[0]:.3f}")
            print(f"\nError words (class 1):")
            print(f"  Precision: {precision[1]:.3f}")
            print(f"  Recall:    {recall[1]:.3f}")
            print(f"  F1:        {f1[1]:.3f}")
            print(f"\nROC AUC: {results[name]['roc_auc']:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
            print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
            
            # Track best model
            if f1[1] > best_f1_error:
                best_f1_error = f1[1]
                best_model_name = name
        
        # Select best model
        self.best_classifier = results[best_model_name]['model']
        self.optimal_threshold = results[best_model_name]['optimal_threshold']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Error class F1: {best_f1_error:.4f}")
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        print('='*60)
        
        return results
    
    def predict(self, audio_path_list, use_joint_features=True, 
                use_optimal_threshold=True, max_length=2):
        """
        Make predictions on new audio files
        
        Args:
            audio_path_list: List of audio paths
            use_joint_features: Whether to use joint features
            use_optimal_threshold: Whether to use optimal threshold
            max_length: Max audio length in seconds
        
        Returns:
            DataFrame with predictions
        """
        if self.best_classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"\nMaking predictions on {len(audio_path_list)} files...")
        if use_optimal_threshold:
            print(f"Using optimal threshold: {self.optimal_threshold:.3f}")
        
        predictions = []
        confidences = []
        successful_paths = []
        
        for audio_path in tqdm(audio_path_list):
            # Load audio
            audio = self.load_audio(audio_path, max_length=max_length)
            if audio is None:
                continue
            
            # Extract features
            if use_joint_features:
                features = self.extract_joint_features(audio)
            else:
                features = self.extract_acoustic_features(audio)
            
            if features is None:
                continue
            
            # Scale and predict
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            if hasattr(self.best_classifier, 'predict_proba'):
                proba = self.best_classifier.predict_proba(features_scaled)[0]
                error_prob = proba[1]
                
                if use_optimal_threshold:
                    prediction = 1 if error_prob >= self.optimal_threshold else 0
                else:
                    prediction = self.best_classifier.predict(features_scaled)[0]
                
                confidence = max(proba)
            else:
                prediction = self.best_classifier.predict(features_scaled)[0]
                confidence = None
            
            predictions.append(prediction)
            confidences.append(confidence)
            successful_paths.append(audio_path)
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'Path': successful_paths,
            'Prediction': predictions,
            'Confidence': confidences
        })
        
        print(f"Successfully predicted: {len(successful_paths)}/{len(audio_path_list)}")
        print(f"Predicted errors: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        
        return results_df
    
    def save_model(self, save_path):
        """Save trained model"""
        model_data = {
            'classifier': self.best_classifier,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold
        }
        joblib.dump(model_data, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path):
        """Load trained model"""
        model_data = joblib.load(load_path)
        self.best_classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.optimal_threshold = model_data['optimal_threshold']
        print(f"Model loaded from: {load_path}")

# ===============================================
# MAIN EXECUTION
# ===============================================

def main():
    parser = argparse.ArgumentParser(
        description='Joint ASR + Error Detection for Child Speech',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_file', type=str, 
                       default='processed_annotations/word_level_data.csv',
                       help='Path to word-level data CSV')
    parser.add_argument('--target_error', type=str, required=True,
                       choices=['Correct', 'Phonological', 'Grammatical', 
                               'Disfluency', 'Orthographic', 'Structural'],
                       help='Target error category to detect')
    parser.add_argument('--use_joint', action='store_true', default=True,
                       help='Use joint ASR+acoustic features')
    parser.add_argument('--acoustic_only', action='store_true',
                       help='Use acoustic features only (overrides --use_joint)')
    parser.add_argument('--max_length', type=float, default=2.0,
                       help='Maximum audio length in seconds')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--cache_dir', type=str, default='audio_features_cache_withASRoutput',
                       help='Directory for feature caching')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine feature type
    use_joint_features = args.use_joint and not args.acoustic_only
    
    # Map error category names
    error_map = {
        'Phonological': 'Phonological Error',
        'Grammatical': 'Grammatical Error',
        'Disfluency': 'Disfluency Error',
        'Orthographic': 'Orthographic Error',
        'Structural': 'Structural Error'
    }
    target_error = error_map.get(args.target_error, args.target_error)
    
    print(f"\n{'='*60}")
    print("JOINT ASR + ERROR DETECTION")
    print('='*60)
    print(f"Target error: {target_error}")
    print(f"Joint features: {use_joint_features}")
    print(f"Max audio length: {args.max_length}s")
    print('='*60)
    
    # Load data
    audio_paths, error_categories = load_and_prepare_data(args.data_file)
    
    # Validate audio files
    print("\nValidating audio files...")
    clean_paths, clean_labels, problem_files = validate_audio_dataset(
        audio_paths, error_categories
    )
    
    if len(clean_paths) == 0:
        print("ERROR: No valid audio files found!")
        return

    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = JointASRErrorClassifier(
        acoustic_model_name="facebook/wav2vec2-large-xlsr-53",
        asr_model_name="facebook/wav2vec2-base-960h",
        cache_dir=args.cache_dir
    )
    
    # Prepare dataset
    feature_type = "joint" if use_joint_features else "acoustic"
    features_path = os.path.join(
        args.output_dir,
        f"{target_error.replace(' ', '_')}_{feature_type}_features_withASRoutput.pkl"
    )
    
    X, y, filenames = classifier.prepare_dataset(
        clean_paths,
        clean_labels,
        target_error_category=target_error,
        max_length=args.max_length,
        use_joint_features=use_joint_features,
        features_save_path=features_path
    )
    
    if len(X) == 0:
        print("ERROR: No features extracted!")
        return
    
    # Train model
    print("\nTraining models...")
    results = classifier.train(X, y, test_size=args.test_size, cv_folds=args.cv_folds)
    
    # Save detailed results
    results_file = os.path.join(
        args.output_dir,
        f"{target_error.replace(' ', '_')}_{feature_type}_results.txt"
    )
    with open(results_file, 'w') as f:
        f.write(f"Joint ASR + Error Detection Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Target error: {target_error}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Total samples: {len(y)}\n")
        f.write(f"Correct (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)\n")
        f.write(f"Error (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)\n")
        f.write(f"Feature dimension: {X.shape[1]}\n\n")
        
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  CV F1: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})\n")
            f.write(f"  Optimal threshold: {result['optimal_threshold']:.4f}\n")
            f.write(f"  Correct class - P: {result['precision_correct']:.4f}, "
                   f"R: {result['recall_correct']:.4f}, F1: {result['f1_correct']:.4f}\n")
            f.write(f"  Error class   - P: {result['precision_error']:.4f}, "
                   f"R: {result['recall_error']:.4f}, F1: {result['f1_error']:.4f}\n")
            f.write(f"  ROC AUC: {result['roc_auc']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Save model
    model_path = os.path.join(
        args.output_dir,
        f"{target_error.replace(' ', '_')}_{feature_type}_model_withASRoutput.pkl"
    )
    classifier.save_model(model_path)
    
    # Make predictions on all data
    print("\nMaking predictions on full dataset...")
    predictions_df = classifier.predict(
        clean_paths,
        use_joint_features=use_joint_features,
        use_optimal_threshold=True,
        max_length=args.max_length
    )
    
    # Save predictions
    predictions_path = os.path.join(
        args.output_dir,
        f"{target_error.replace(' ', '_')}_{feature_type}_predictions.csv"
    )
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    print(f"Best model: {[k for k, v in results.items() if v['model'] == classifier.best_classifier][0]}")
    print(f"Error F1: {max(r['f1_error'] for r in results.values()):.4f}")
    print(f"Correct F1: {max(r['f1_correct'] for r in results.values()):.4f}")
    print(f"Optimal threshold: {classifier.optimal_threshold:.4f}")
    print('='*60)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_file}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
