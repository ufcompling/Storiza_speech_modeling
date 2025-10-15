#!/usr/bin/env python3

"""
Evaluate a trained Joint ASR + Error Detection model
"""

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import joblib
from tqdm import tqdm
import librosa

def load_audio(audio_path, target_sr=16000, max_length=2):
    """Load audio file"""
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

def evaluate_saved_model(model_path, test_csv, output_path, max_length=2):
    """
    Evaluate a saved model on test data
    
    Args:
        model_path: Path to saved model (.pkl)
        test_csv: Path to CSV with columns: Path, Error Category
        output_path: Where to save predictions
        max_length: Max audio length in seconds
    """
    print(f"Loading model from: {model_path}")
    
    # Check if this is the full classifier or just model data
    model_data = joblib.load(model_path)
    
    if 'classifier' in model_data:
        # It's just the model weights, need to recreate classifier
        print("Model contains classifier weights only")
        print("Recreating JointASRErrorClassifier...")
        
        # Import the classifier class
        from __main__ import JointASRErrorClassifier
        
        classifier = JointASRErrorClassifier()
        classifier.best_classifier = model_data['classifier']
        classifier.scaler = model_data['scaler']
        classifier.optimal_threshold = model_data.get('optimal_threshold', 0.5)
    else:
        # It's the full classifier object
        classifier = model_data
    
    print(f"Optimal threshold: {classifier.optimal_threshold:.3f}")
    
    # Load test data
    print(f"\nLoading test data from: {test_csv}")
    test_data = pd.read_csv(test_csv)
    
    audio_paths = test_data['Path'].tolist()
    
    # Check if we have ground truth labels
    has_labels = 'Error Category' in test_data.columns
    if has_labels:
        error_categories = test_data['Error Category'].tolist()
        print(f"Test samples: {len(audio_paths)} (with ground truth)")
    else:
        print(f"Test samples: {len(audio_paths)} (no ground truth)")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    confidences = []
    successful_paths = []
    
    for audio_path in tqdm(audio_paths):
        audio = load_audio(audio_path, max_length=max_length)
        if audio is None:
            continue
        
        # Extract features - assume joint features were used during training
        features = classifier.extract_joint_features(audio)
        if features is None:
            continue
        
        # Scale and predict
        features_scaled = classifier.scaler.transform(features.reshape(1, -1))
        
        if hasattr(classifier.best_classifier, 'predict_proba'):
            proba = classifier.best_classifier.predict_proba(features_scaled)[0]
            error_prob = proba[1]
            prediction = 1 if error_prob >= classifier.optimal_threshold else 0
            confidence = max(proba)
        else:
            prediction = classifier.best_classifier.predict(features_scaled)[0]
            confidence = None
        
        predictions.append(prediction)
        confidences.append(confidence)
        successful_paths.append(audio_path)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Path': successful_paths,
        'Prediction': predictions,
        'Confidence': confidences
    })
    
    # Calculate metrics if we have ground truth
    if has_labels:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        
        # Get ground truth for successful predictions
        true_labels = []
        for path in successful_paths:
            idx = audio_paths.index(path)
            # Determine if it's an error (you'll need to specify target error)
            # For now, assume binary: 1 if any error, 0 if Correct
            categories = error_categories[idx]
            if isinstance(categories, str):
                categories = eval(categories)  # Convert string representation of list
            
            # Simple: 1 if not "Correct", 0 otherwise
            has_error = int('Correct' not in categories if isinstance(categories, list) else categories != 'Correct')
            true_labels.append(has_error)
        
        results_df['Ground Truth'] = true_labels
        
        # Print metrics
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(classification_report(true_labels, predictions, 
                                   target_names=['Correct', 'Error']))
        
        cm = confusion_matrix(true_labels, predictions)
        print("\nConfusion Matrix:")
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        # Per-class F1
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        print(f"\nPer-class metrics:")
        print(f"  Correct - P: {precision[0]:.3f}, R: {recall[0]:.3f}, F1: {f1[0]:.3f}")
        print(f"  Error   - P: {precision[1]:.3f}, R: {recall[1]:.3f}, F1: {f1[1]:.3f}")
    
    # Save predictions
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"Successfully predicted: {len(successful_paths)}/{len(audio_paths)}")
    print(f"Predicted errors: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained Joint ASR + Error Detection model'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (.pkl)')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                       help='Where to save predictions')
    parser.add_argument('--max_length', type=float, default=2.0,
                       help='Maximum audio length in seconds')
    
    args = parser.parse_args()
    
    evaluate_saved_model(
        args.model_path,
        args.test_csv,
        args.output_path,
        args.max_length
    )


if __name__ == "__main__":
    main()