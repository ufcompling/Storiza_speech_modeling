#!/usr/bin/env python3

"""
Audio Binary Classification using Wav2Vec2 Features
Extracts features from audio files using pre-trained wav2vec2 model
and trains statistical classifiers for binary classification.
"""

import os, sys
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from tqdm import tqdm
import random
import hashlib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

word_segments_data_file = 'processed_annotations/word_level_data.csv'
full_word_segments_data = pd.read_csv(word_segments_data_file)

word_segments_data = full_word_segments_data.sample(frac=1)
audio_path_list = word_segments_data['Path'].tolist()

word_segments_data["Error Category"] = word_segments_data["Error Category"].apply(
    lambda x: [category.strip() for category in x.split("+") if category != 'Mixed Error']
)
error_category_list = word_segments_data['Error Category'].tolist()

word_segments_data['Error Labels'] = word_segments_data['Error Labels'].apply(
    lambda labels: labels.strip('[]').split(', ') if labels != '[]' else ['NONE']
)
error_label_list = word_segments_data['Error Labels'].tolist()

# Consolidate error categories
error_map = {
    'Grammatical': 'Grammatical Error',
    'Orthographic Sub.': 'Orthographic Error',
    'Phonological': 'Phonological Error',
    'Run-on': 'Correct',
    'Structural': 'Structural Error',
    'Visual Tracking': 'Visual Tracking Error',
    'Contraction/Shortening': 'Correct'
}

modified_error_category_list = []
for i in range(len(error_category_list)):
    error_categories = error_category_list[i]
    for k, v in error_map.items():
        while k in error_categories:
            error_categories = [error_map[category] if category in error_map else category 
                              for category in error_categories]
    modified_error_category_list.append(error_categories)

# Print error category distribution
error_category_dist = {}
for i in range(len(modified_error_category_list)):
    categories = modified_error_category_list[i]
    for category in categories:
        if category in error_category_dist:
            error_category_dist[category] += 1
        else:
            error_category_dist[category] = 1

sorted_error_category_dist = dict(sorted(error_category_dist.items(), 
                                        key=lambda item: item[1], reverse=True))
print("Error Category Distribution")
for category, count in sorted_error_category_dist.items():
    print(f"{category}: {count} / {count/sum(sorted_error_category_dist.values()):.2%}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_binary_labels(target_error_category, modified_error_category_list):
    """Create binary labels based on target error category"""
    binary_labels = []
    for categories in modified_error_category_list:
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
            if not os.path.exists(path):
                problem_files.append((path, "File not found"))
                continue
            
            file_size = os.path.getsize(path)
            if file_size == 0:
                problem_files.append((path, "Empty file (0 bytes)"))
                continue
            
            try:
                waveform, sample_rate = torchaudio.load(path)
            except Exception as e:
                problem_files.append((path, f"Cannot load audio: {str(e)}"))
                continue
            
            if waveform.numel() == 0:
                problem_files.append((path, "No audio data"))
                continue
            
            duration = waveform.shape[-1] / sample_rate
            if duration < min_duration:
                problem_files.append((path, f"Too short: {duration:.3f}s"))
                continue
            
            if torch.all(waveform == 0):
                problem_files.append((path, "Silent audio (all zeros)"))
                continue
            
            clean_paths.append(path)
            clean_labels.append(label)
            
        except Exception as e:
            problem_files.append((path, f"Validation error: {str(e)}"))
        
        if (i + 1) % 100 == 0:
            print(f"Validated {i + 1}/{len(audio_paths)} files...")
    
    print(f"\nDataset Validation Summary:")
    print(f"✅ Valid files: {len(clean_paths)}")
    print(f"❌ Problem files: {len(problem_files)}")
    
    if problem_files:
        with open('problem_files.txt', 'w') as f:
            for path, issue in problem_files:
                f.write(f"{path}: {issue}\n")
    
    return clean_paths, clean_labels, problem_files

# =============================================================================
# AUDIO CLASSIFIER CLASSES
# =============================================================================

class AudioClassifier:
    """Base audio classifier with wav2vec2 feature extraction"""
    
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53"):
        print("Initializing AudioClassifier...")
        print(f"Model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=8, 
                                                   class_weight='balanced'),
            'svm': SVC(kernel='rbf', random_state=8, class_weight='balanced', probability=True),
            'xgboost': xgb.XGBClassifier(random_state=8, eval_metric='logloss')
        }
        
        self.scaler = StandardScaler()
        self.best_classifier = None
        
    def load_audio(self, audio_path, target_sr=16000, max_length=None):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            if max_length is not None:
                max_samples = int(max_length * target_sr)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_wav2vec2_features(self, audio):
        """Extract features using wav2vec2 model"""
        try:
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                features = torch.mean(hidden_states, dim=1).squeeze()
                features = features.cpu().numpy()
                
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def train(self, X, y, test_size=0.2, cv_folds=2, X_test_original=None, y_test_original=None):
        """
        Train multiple classifiers and select the best one
        
        Args:
            X: Feature matrix (possibly augmented)
            y: Labels
            test_size: Proportion of test set (only used if X_test_original is None)
            cv_folds: Number of cross-validation folds
            X_test_original: Optional separate test set with original data only
            y_test_original: Labels for original test set
        """
        # Split data or use provided test set
        if X_test_original is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=8, stratify=y
            )
        else:
            X_train = X
            y_train = y
            X_test = X_test_original
            y_test = y_test_original
            print(f"Using separate original test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training classifiers...")
        results = {}
        
        for name, classifier in self.classifiers.items():
            print(f"\nTraining {name}...")
            
            cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=cv_folds)
            classifier.fit(X_train_scaled, y_train)
            test_score = classifier.score(X_test_scaled, y_test)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'model': classifier
            }
            
            print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"Test Score: {test_score:.3f}")
        
        best_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_classifier = results[best_name]['model']
        
        print(f"\nBest classifier: {best_name}")
        print(f"Best CV score: {results[best_name]['cv_mean']:.3f}")
        
        y_pred = self.best_classifier.predict(X_test_scaled)
        test_type = 'ORIGINAL' if X_test_original is not None else 'augmented'
        print(f"\nDetailed evaluation of {best_name} on {test_type} test set:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return results
    
    def predict(self, audio_path_list):
        """Predict class for a list of audio files"""
        prediction_list = []
        confidence_list = []
        successful_paths = []

        if self.best_classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"Making predictions for {len(audio_path_list)} audio files...")
        
        for audio_file in tqdm(audio_path_list):
            audio = self.load_audio(audio_file)
            if audio is None:
                continue
            
            features = self.extract_wav2vec2_features(audio)
            if features is None:
                continue
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.best_classifier.predict(features_scaled)[0]
            
            if hasattr(self.best_classifier, 'predict_proba'):
                probabilities = self.best_classifier.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = None
            
            prediction_list.append(prediction)
            confidence_list.append(confidence)
            successful_paths.append(audio_file)
        
        prediction_confidence_data = pd.DataFrame({
            'Path': successful_paths,
            'Prediction': prediction_list,
            'Confidence': confidence_list
        })
        
        csv_filename = sys.argv[1] + '_prediction_augment.csv'
        prediction_confidence_data.to_csv(csv_filename, index=False)
        
        print(f"Predictions completed!")
        print(f"Successfully processed: {len(successful_paths)} out of {len(audio_path_list)} files")
        print(f"Results saved to: {csv_filename}")
        
        return prediction_confidence_data
    
    def save_model(self, save_path):
        """Save the trained model and scaler"""
        model_data = {
            'classifier': self.best_classifier,
            'scaler': self.scaler,
            'processor_name': 'wav2vec2-large-xlsr-53',
            'model_name': 'wav2vec2-large-xlsr-53'
        }
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        model_data = joblib.load(model_path)
        self.best_classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        print(f"Model loaded from {model_path}")


class AudioClassifierWithCaching(AudioClassifier):
    """Extended AudioClassifier with feature caching capabilities"""
    
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", cache_dir="audio_augment_features_cache"):
        super().__init__(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, audio_path, max_length):
        """Generate unique cache key for audio file"""
        file_stat = os.stat(audio_path)
        key_string = f"{audio_path}_{file_stat.st_mtime}_{max_length}_wav2vec2-large-xlsr-53"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_features_to_cache(self, cache_key, features):
        """Save features to cache file"""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_file, features)
    
    def _load_features_from_cache(self, cache_key):
        """Load features from cache file"""
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None
    
    def extract_wav2vec2_features_cached(self, audio_path, max_length=None):
        """Extract features with caching support"""
        cache_key = self._get_cache_key(audio_path, max_length)
        cached_features = self._load_features_from_cache(cache_key)
        
        if cached_features is not None:
            return cached_features
        
        audio = self.load_audio(audio_path, max_length=max_length)
        if audio is None:
            return None
            
        features = self.extract_wav2vec2_features(audio)
        if features is not None:
            self._save_features_to_cache(cache_key, features)
        
        return features
    
    def prepare_dataset_cached(self, audio_path_list, modified_error_category_list, 
                              target_error_category='Correct', max_length=30, 
                              features_save_path=None):
        """Prepare dataset with feature caching"""
        assert len(audio_path_list) == len(modified_error_category_list), \
            "Audio paths and labels must be of same length"
        
        if features_save_path and os.path.exists(features_save_path):
            print(f"Loading pre-saved features from {features_save_path}")
            return self.load_features_dataset(features_save_path)
        
        features_list = []
        labels_list = []
        audio_filenames = []
        binary_labels_list = create_binary_labels(target_error_category, modified_error_category_list)
        
        print(f"Processing {len(audio_path_list)} audio files with caching...")
        
        cache_hits = 0
        cache_misses = 0
        
        for i, audio_file in enumerate(tqdm(audio_path_list)):
            cache_key = self._get_cache_key(audio_file, max_length)
            features = self._load_features_from_cache(cache_key)
            
            if features is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                features = self.extract_wav2vec2_features_cached(audio_file, max_length)
                if features is None:
                    continue
            
            features_list.append(features)
            labels_list.append(binary_labels_list[i])
            audio_filenames.append(Path(audio_file).name)
        
        print(f"Cache performance: {cache_hits} hits, {cache_misses} misses")
        print(f"Successfully processed {len(features_list)} files")
        
        if features_save_path:
            self.save_features_dataset(features_list, labels_list, audio_filenames, features_save_path)
        
        return np.array(features_list), np.array(labels_list), audio_filenames
    
    def save_features_dataset(self, features_list, labels_list, filenames, save_path):
        """Save entire feature dataset to disk"""
        dataset = {
            'features': np.array(features_list),
            'labels': np.array(labels_list),
            'filenames': filenames,
            'model_name': 'wav2vec2-large-xlsr-53',
            'feature_dim': features_list[0].shape[0] if features_list else None
        }
        joblib.dump(dataset, save_path)
        print(f"Features dataset saved to {save_path}")
        print(f"Dataset info: {len(features_list)} samples, {dataset['feature_dim']} features")
    
    def load_features_dataset(self, load_path):
        """Load entire feature dataset from disk"""
        dataset = joblib.load(load_path)
        print(f"Features dataset loaded from {load_path}")
        print(f"Dataset info: {len(dataset['features'])} samples, {dataset['feature_dim']} features")
        return dataset['features'], dataset['labels'], dataset['filenames']


class AudioAugmenter:
    """Audio augmentation methods optimized for speech data"""
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        
    def add_noise(self, audio, noise_factor=0.005):
        """Add white noise"""
        noise = np.random.normal(0, noise_factor, len(audio))
        noise = np.clip(noise, -0.1, 0.1)
        augmented = audio + noise
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val * 0.95
        return augmented
    
    def time_stretch(self, audio, rate_range=(0.85, 1.15)):
        """Change playback speed without changing pitch"""
        rate = random.uniform(*rate_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched
    
    def pitch_shift(self, audio, n_steps_range=(-1.5, 1.5)):
        """Shift pitch without changing duration"""
        n_steps = random.uniform(*n_steps_range)
        shifted = librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=n_steps)
        return shifted
    
    def gain_change(self, audio, gain_range=(0.8, 1.2)):
        """Change volume/gain of audio"""
        gain = random.uniform(*gain_range)
        augmented = audio * gain
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / max_val * 0.95
        return augmented
    
    def time_shift(self, audio, shift_range=(-0.1, 0.1)):
        """Shift audio in time (circular shift)"""
        shift_samples = int(random.uniform(*shift_range) * len(audio))
        if shift_samples != 0:
            augmented = np.roll(audio, shift_samples)
        else:
            augmented = audio.copy()
        return augmented


class AudioClassifierWithCachingAndAugmentation(AudioClassifierWithCaching):
    """Complete audio classifier with caching and augmentation"""
    
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", cache_dir="audio_features_cache"):
        super().__init__(model_name, cache_dir)
        self.augmenter = AudioAugmenter(target_sr=16000)
        
    def apply_augmentations(self, audio, n_augmentations=2):
        """Apply random augmentations to audio"""
        augmented = audio.copy()
        
        augmentations = [
            (self.augmenter.add_noise, 0.6, {'noise_factor': random.uniform(0.002, 0.008)}),
            (self.augmenter.time_stretch, 0.4, {}),
            (self.augmenter.pitch_shift, 0.3, {}),
            (self.augmenter.gain_change, 0.5, {}),
            (self.augmenter.time_shift, 0.3, {})
        ]
        
        selected = random.sample(augmentations, min(n_augmentations, len(augmentations)))
        
        for aug_func, prob, kwargs in selected:
            if random.random() < prob:
                try:
                    augmented = aug_func(augmented, **kwargs)
                except Exception:
                    continue
        
        return augmented
    
    def _get_augmented_cache_key(self, audio_path, max_length, aug_idx):
        """Generate cache key for augmented audio"""
        file_stat = os.stat(audio_path)
        key_string = f"{audio_path}_{file_stat.st_mtime}_{max_length}_aug_{aug_idx}_wav2vec2-large-xlsr-53"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def extract_features_with_augmentation(self, audio_path, max_length=30, n_augmentations=0):
        """Extract features for original + augmented versions"""
        features_list = []
        
        # Original audio features
        original_features = self.extract_wav2vec2_features_cached(audio_path, max_length)
        if original_features is not None:
            features_list.append(original_features)
        
        # Augmented audio features
        for aug_idx in range(n_augmentations):
            aug_cache_key = self._get_augmented_cache_key(audio_path, max_length, aug_idx)
            cached_aug_features = self._load_features_from_cache(aug_cache_key)
            
            if cached_aug_features is not None:
                features_list.append(cached_aug_features)
            else:
                audio = self.load_audio(audio_path, max_length=max_length)
                if audio is not None:
                    augmented_audio = self.apply_augmentations(audio)
                    aug_features = self.extract_wav2vec2_features(augmented_audio)
                    if aug_features is not None:
                        self._save_features_to_cache(aug_cache_key, aug_features)
                        features_list.append(aug_features)
        
        return features_list
    
    def prepare_dataset_with_augmentation(self, audio_path_list, modified_error_category_list, 
                                        target_error_category='Correct', max_length=30,
                                        augmentation_factor=1, balance_classes=True,
                                        features_save_path=None):
        """Prepare dataset with intelligent augmentation and caching"""
        assert len(audio_path_list) == len(modified_error_category_list), "Paths and labels must match"
        
        if features_save_path and os.path.exists(features_save_path):
            print(f"Loading pre-saved augmented features from {features_save_path}")
            return self.load_features_dataset(features_save_path)
        
        features_list = []
        labels_list = []
        filenames_list = []
        
        binary_labels_list = create_binary_labels(target_error_category, modified_error_category_list)
        
        unique_labels, counts = np.unique(binary_labels_list, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        minority_class = unique_labels[np.argmin(counts)]
        
        print('\n')
        print("Classifying " + target_error_category)
        print(f"Original class distribution: {dict(zip(unique_labels, counts))}")
        print(f"Processing {len(audio_path_list)} files with augmentation...")
        print('\n')

        for audio_path, label in tqdm(zip(audio_path_list, binary_labels_list), 
                                      total=len(audio_path_list)):
            
            if balance_classes and label == minority_class:
                n_augmentations = augmentation_factor * 3
            else:
                n_augmentations = augmentation_factor
            
            all_features = self.extract_features_with_augmentation(
                audio_path, max_length, n_augmentations
            )
            
            for features in all_features:
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                    filenames_list.append(f"{Path(audio_path).stem}_v{len(features_list)}")
        
        print(f"Generated {len(features_list)} total samples from {len(audio_path_list)} original files")
        
        unique_aug_labels, aug_counts = np.unique(labels_list, return_counts=True)
        print(f"Augmented class distribution: {dict(zip(unique_aug_labels, aug_counts))}")
        
        if features_save_path:
            self.save_features_dataset(features_list, labels_list, filenames_list, features_save_path)
        
        return np.array(features_list), np.array(labels_list), filenames_list


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(target_error_category):
    """Main function with separate original test set evaluation"""
    
    # Validate dataset
    print("Validating audio dataset...")
    clean_paths, clean_labels, problem_files = validate_audio_dataset(
        audio_path_list, modified_error_category_list
    )
    
    print("\n" + "=" * 70)
    print("STEP 1: Creating train/test split from ORIGINAL data")
    print("=" * 70)
    
    if target_error_category == 'Disfluency':
        target_error_category = 'Disfluency Error'
    if target_error_category == 'Grammatical':
        target_error_category = 'Grammatical Error'
    if target_error_category == 'Phonological':
        target_error_category = 'Phonological Error'
    binary_labels_original = create_binary_labels(target_error_category, clean_labels)
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        clean_paths, 
        clean_labels, 
        test_size=0.2, 
        random_state=8, 
        stratify=binary_labels_original
    )
    
    print(f"Original train set: {len(train_paths)} files")
    print(f"Original test set: {len(test_paths)} files")
    
    classifier = AudioClassifierWithCachingAndAugmentation(
        model_name="facebook/wav2vec2-large-xlsr-53"
    )
    
    print("\n" + "=" * 70)
    print("STEP 2: Extract features from ORIGINAL TEST SET (no augmentation)")
    print("=" * 70)
    
    X_test_original, y_test_original, test_filenames = classifier.prepare_dataset_cached(
        test_paths,
        test_labels,
        target_error_category=target_error_category,
        max_length=15,
        features_save_path=target_error_category.replace(' ', '_') + '_test_features.pkl'
    )
    
    print(f"\nOriginal test set class distribution:")
    unique, counts = np.unique(y_test_original, return_counts=True)
    print(dict(zip(unique, counts)))
    
    print("\n" + "=" * 70)
    print("STEP 3: Augment TRAINING SET ONLY")
    print("=" * 70)
    
    X_train_augmented, y_train_augmented, train_filenames = classifier.prepare_dataset_with_augmentation(
        train_paths, 
        train_labels, 
        target_error_category=target_error_category,
        augmentation_factor=1,
        max_length=15,
        features_save_path=target_error_category.replace(' ', '_') + '_augmented_train_features.pkl'
    )
    
    print("\n" + "=" * 70)
    print("STEP 4: Train on AUGMENTED data, test on ORIGINAL data")
    print("=" * 70)
    
    results = classifier.train(
        X_train_augmented, 
        y_train_augmented,
        X_test_original=X_test_original,
        y_test_original=y_test_original
    )
    
    print("\n" + "=" * 70)
    print("STEP 5: Save model and make predictions")
    print("=" * 70)
    
    classifier.save_model(target_error_category.replace(' ', '_') + '_augmented_model.pkl')
    
    print("\nMaking predictions on original test set only...")
    predictions = classifier.predict(test_paths)
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Training set size (augmented): {len(X_train_augmented)} samples")
    print(f"Test set size (original only): {len(X_test_original)} samples")
    print(f"Model trained on augmented data, evaluated on clean original data")
    print("=" * 70)
    
    return classifier, results


if __name__ == "__main__":
    main(sys.argv[1])
