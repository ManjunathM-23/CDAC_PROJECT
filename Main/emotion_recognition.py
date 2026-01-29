import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import glob
import pyaudio
import wave
import threading
import queue
import noisereduce as nr
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# ===== AUDIO PREPROCESSING =====

def apply_noise_reduction(audio, sr):
    """Apply noise reduction to audio"""
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

def butter_bandpass(lowcut, highcut, sr, order=5):
    """Create butterworth bandpass filter"""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(audio, lowcut=300, highcut=3400, sr=22050, order=5):
    """Apply bandpass filter (simulates telephone quality)"""
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    return lfilter(b, a, audio)

def preprocess_audio(file_path, apply_noise_red=True, apply_bandpass=True):
    """Load and preprocess audio file"""
    audio, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Apply noise reduction
    if apply_noise_red:
        audio = apply_noise_reduction(audio, sr)
    
    # Apply bandpass filter
    if apply_bandpass:
        audio = bandpass_filter(audio, sr=sr)
    
    return audio, sr

# ===== DATA AUGMENTATION =====

def add_noise(audio, noise_factor=0.005):
    """Add random noise to audio"""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented / np.max(np.abs(augmented))

def pitch_shift(audio, sr, n_steps=2):
    """Shift pitch of audio"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=1.2):
    """Stretch time of audio"""
    return librosa.effects.time_stretch(audio, rate=rate)

def change_speed(audio, speed_factor=1.1):
    """Change speed of audio"""
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def augment_audio(audio, sr):
    """Apply random augmentation to audio"""
    augmentations = []
    
    # Original
    augmentations.append(audio)
    
    # Add noise (small and large)
    augmentations.append(add_noise(audio, 0.003))
    augmentations.append(add_noise(audio, 0.007))
    
    # Pitch shift (up and down)
    augmentations.append(pitch_shift(audio, sr, 2))
    augmentations.append(pitch_shift(audio, sr, -2))
    
    # Time stretch
    augmentations.append(time_stretch(audio, 0.9))
    augmentations.append(time_stretch(audio, 1.1))
    
    # Speed change
    augmentations.append(change_speed(audio, 0.95))
    augmentations.append(change_speed(audio, 1.05))
    
    return augmentations

# ===== FEATURE EXTRACTION =====

def extract_features(audio, sr, mfcc=True, chroma=True, mel=True, 
                    contrast=True, tonnetz=True, zcr=True):
    """Extract comprehensive audio features"""
    result = np.array([])
    
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))
    
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        result = np.hstack((result, mel))
    
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        result = np.hstack((result, contrast))
    
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
        result = np.hstack((result, tonnetz))
    
    if zcr:
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
        result = np.hstack((result, zcr))
    
    return result

def extract_features_from_file(file_path, augment=False):
    """Extract features from file with optional augmentation"""
    audio, sr = preprocess_audio(file_path)
    
    if not augment:
        return [extract_features(audio, sr)]
    else:
        augmented_audios = augment_audio(audio, sr)
        return [extract_features(aug_audio, sr) for aug_audio in augmented_audios]

# ===== MULTI-DATASET LOADING =====

def load_ravdess_data(data_path, augment=False):
    """Load RAVDESS dataset"""
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    X, y = [], []
    
    for file in glob.glob(os.path.join(data_path, "Actor_*/*.wav")):
        emotion_code = os.path.basename(file).split("-")[2]
        emotion = emotions.get(emotion_code)
        
        if emotion:
            try:
                features_list = extract_features_from_file(file, augment)
                for features in features_list:
                    X.append(features)
                    y.append(emotion)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    return X, y

def load_tess_data(data_path, augment=False):
    """Load TESS dataset"""
    emotions_map = {
        'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
        'happy': 'happy', 'neutral': 'neutral', 'ps': 'surprised', 'sad': 'sad'
    }
    
    X, y = [], []
    
    for emotion_folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, emotion_folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Extract emotion from folder name
        emotion_key = emotion_folder.lower().split('_')[-1]
        emotion = emotions_map.get(emotion_key)
        
        if emotion:
            for file in glob.glob(os.path.join(folder_path, "*.wav")):
                try:
                    features_list = extract_features_from_file(file, augment)
                    for features in features_list:
                        X.append(features)
                        y.append(emotion)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    return X, y

def load_crema_data(data_path, augment=False):
    """Load CREMA-D dataset"""
    # CREMA-D emotion mapping
    emotions_map = {
        'ANG': 'angry',     # Anger
        'DIS': 'disgust',   # Disgust
        'FEA': 'fearful',   # Fear
        'HAP': 'happy',     # Happy
        'NEU': 'neutral',   # Neutral
        'SAD': 'sad'        # Sadness
    }
    
    X, y = [], []
    
    # Handle both direct path and AudioWAV subfolder
    wav_paths = [
        os.path.join(data_path, "*.wav"),
        os.path.join(data_path, "AudioWAV", "*.wav")
    ]
    
    files_found = 0
    for wav_path in wav_paths:
        for file in glob.glob(wav_path):
            files_found += 1
            # Extract emotion from filename
            # Format: 1001_DFA_ANG_XX.wav or 1001_IEO_FEA_LO.wav
            filename = os.path.basename(file)
            parts = filename.replace('.wav', '').split("_")
            
            if len(parts) >= 3:
                emotion_code = parts[2]  # Third part is emotion (ANG, DIS, FEA, etc.)
                emotion = emotions_map.get(emotion_code)
                
                if emotion:
                    try:
                        features_list = extract_features_from_file(file, augment)
                        for features in features_list:
                            X.append(features)
                            y.append(emotion)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                else:
                    if files_found <= 5:  # Only print first few unknown codes
                        print(f"Unknown emotion code '{emotion_code}' in file: {filename}")
    
    if files_found == 0:
        print(f"Warning: No .wav files found in {data_path}")
        print(f"Please check if the path is correct or if files are in an AudioWAV subfolder")
    
    return X, y

def load_all_datasets(ravdess_path=None, tess_path=None, crema_path=None, augment=True):
    """Load and combine multiple datasets"""
    X_all, y_all = [], []
    
    if ravdess_path and os.path.exists(ravdess_path):
        print("Loading RAVDESS dataset...")
        X, y = load_ravdess_data(ravdess_path, augment)
        X_all.extend(X)
        y_all.extend(y)
        print(f"RAVDESS: {len(X)} samples loaded")
    
    if tess_path and os.path.exists(tess_path):
        print("Loading TESS dataset...")
        X, y = load_tess_data(tess_path, augment)
        X_all.extend(X)
        y_all.extend(y)
        print(f"TESS: {len(X)} samples loaded")
    
    if crema_path and os.path.exists(crema_path):
        print("Loading CREMA-D dataset...")
        X, y = load_crema_data(crema_path, augment)
        X_all.extend(X)
        y_all.extend(y)
        print(f"CREMA-D: {len(X)} samples loaded")
    
    print(f"\nTotal samples: {len(X_all)}")
    return np.array(X_all), np.array(y_all)

# ===== MODEL ARCHITECTURE WITH HYPERPARAMETERS =====

def create_cnn_lstm_model(input_shape, num_classes, config):
    """Create CNN + LSTM model with configurable hyperparameters"""
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        
        # CNN Block 1
        layers.Conv1D(config['cnn_filters'][0], config['kernel_size'], 
                     padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(config['dropout_cnn']),
        
        # CNN Block 2
        layers.Conv1D(config['cnn_filters'][1], config['kernel_size'], 
                     padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(config['dropout_cnn']),
        
        # CNN Block 3
        layers.Conv1D(config['cnn_filters'][2], config['kernel_size'], 
                     padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(config['dropout_cnn']),
        
        # LSTM layers
        layers.LSTM(config['lstm_units'][0], return_sequences=True),
        layers.Dropout(config['dropout_lstm']),
        layers.LSTM(config['lstm_units'][1]),
        layers.Dropout(config['dropout_lstm']),
        
        # Dense layers
        layers.Dense(config['dense_units'][0], activation='relu'),
        layers.Dropout(config['dropout_dense']),
        layers.Dense(config['dense_units'][1], activation='relu'),
        layers.Dropout(config['dropout_dense']),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ===== TRAINING WITH HYPERPARAMETER CONFIGURATIONS =====

def get_default_config():
    """Get default hyperparameter configuration"""
    return {
        'cnn_filters': [64, 128, 256],
        'kernel_size': 5,
        'lstm_units': [128, 64],
        'dense_units': [128, 64],
        'dropout_cnn': 0.3,
        'dropout_lstm': 0.3,
        'dropout_dense': 0.4,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 100
    }

def get_optimized_config():
    """Get optimized hyperparameter configuration"""
    return {
        'cnn_filters': [128, 256, 512],
        'kernel_size': 7,
        'lstm_units': [256, 128],
        'dense_units': [256, 128],
        'dropout_cnn': 0.4,
        'dropout_lstm': 0.4,
        'dropout_dense': 0.5,
        'learning_rate': 0.00005,
        'batch_size': 64,
        'epochs': 150
    }

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def train_model(ravdess_path=None, tess_path=None, crema_path=None, 
                model_save_path='emotion_model.h5', config=None, augment=True):
    """Train the emotion recognition model"""
    
    if config is None:
        config = get_default_config()
    
    print("Loading datasets...")
    X, y = load_all_datasets(ravdess_path, tess_path, crema_path, augment)
    
    if len(X) == 0:
        print("Error: No data loaded. Please check dataset paths.")
        return None, None, None
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Feature shape: {X_train.shape[1]}")
    print(f"Emotion classes: {le.classes_}")
    
    # Create model
    model = create_cnn_lstm_model((X_train.shape[1],), y_categorical.shape[1], config)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=20, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=7, 
            min_lr=1e-7,
            monitor='val_loss'
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path, 
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=le.classes_))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test_classes, y_pred_classes, le.classes_)
    
    # Save artifacts
    np.save('label_encoder_classes.npy', le.classes_)
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nTraining complete! Model saved to", model_save_path)
    
    return model, history, le

# ===== REAL-TIME PREDICTION =====

class RealTimeEmotionRecognizer:
    """Real-time emotion recognition from microphone"""
    
    def __init__(self, model_path, label_classes_path):
        self.model = keras.models.load_model(model_path)
        self.label_classes = np.load(label_classes_path, allow_pickle=True)
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 3
        
        self.audio = pyaudio.PyAudio()
        self.emotion_history = []
    
    def record_audio(self):
        """Record audio from microphone"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("🎤 Recording...")
        frames = []
        
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        print("✓ Recording finished")
        
        return b''.join(frames)
    
    def predict_emotion(self, audio_data):
        """Predict emotion from audio data"""
        temp_file = "temp_audio.wav"
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(audio_data)
        wf.close()
        
        try:
            # Preprocess and extract features
            audio, sr = preprocess_audio(temp_file)
            features = extract_features(audio, sr)
            
            # Predict
            features = np.expand_dims(features, axis=0)
            prediction = self.model.predict(features, verbose=0)
            emotion_idx = np.argmax(prediction)
            confidence = prediction[0][emotion_idx]
            emotion = self.label_classes[emotion_idx]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            top_3_emotions = [(self.label_classes[i], prediction[0][i]) 
                             for i in top_3_idx]
            
            os.remove(temp_file)
            return emotion, confidence, top_3_emotions
        
        except Exception as e:
            print(f"Error: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return None, None, None
    
    def start_real_time_recognition(self, show_top_3=True):
        """Start continuous emotion recognition"""
        print("\n" + "="*50)
        print("🎭 Real-Time Emotion Recognition Started")
        print("="*50)
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                audio_data = self.record_audio()
                emotion, confidence, top_3 = self.predict_emotion(audio_data)
                
                if emotion:
                    self.emotion_history.append(emotion)
                    
                    print(f"\n{'='*50}")
                    print(f"🎯 Detected Emotion: {emotion.upper()}")
                    print(f"📊 Confidence: {confidence:.2%}")
                    
                    if show_top_3:
                        print(f"\n📈 Top 3 Predictions:")
                        for i, (em, conf) in enumerate(top_3, 1):
                            bar = '█' * int(conf * 20)
                            print(f"   {i}. {em.capitalize():<12} {conf:.2%} {bar}")
                    
                    print(f"{'='*50}\n")
                else:
                    print("❌ Could not process audio\n")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping recognition...")
            self.print_session_summary()
        finally:
            self.audio.terminate()
    
    def print_session_summary(self):
        """Print summary of emotion detection session"""
        if self.emotion_history:
            print(f"\n{'='*50}")
            print("📊 Session Summary")
            print(f"{'='*50}")
            unique, counts = np.unique(self.emotion_history, return_counts=True)
            for emotion, count in zip(unique, counts):
                percentage = count / len(self.emotion_history) * 100
                print(f"{emotion.capitalize():<12}: {count:>3} ({percentage:>5.1f}%)")
            print(f"{'='*50}\n")

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                       help='Mode: train or predict')
    parser.add_argument('--ravdess_path', type=str, help='Path to RAVDESS dataset')
    parser.add_argument('--tess_path', type=str, help='Path to TESS dataset')
    parser.add_argument('--crema_path', type=str, help='Path to CREMA-D dataset')
    parser.add_argument('--model_path', type=str, default='emotion_model.h5',
                       help='Path to model file')
    parser.add_argument('--config', choices=['default', 'optimized'], 
                       default='default', help='Hyperparameter configuration')
    parser.add_argument('--no_augment', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not any([args.ravdess_path, args.tess_path, args.crema_path]):
            print("Error: At least one dataset path required for training")
            print("Use --ravdess_path, --tess_path, or --crema_path")
        else:
            config = get_optimized_config() if args.config == 'optimized' else get_default_config()
            print(f"\nUsing {args.config} configuration:")
            print(json.dumps(config, indent=2))
            
            model, history, le = train_model(
                ravdess_path=args.ravdess_path,
                tess_path=args.tess_path,
                crema_path=args.crema_path,
                model_save_path=args.model_path,
                config=config,
                augment=not args.no_augment
            )
    
    elif args.mode == 'predict':
        recognizer = RealTimeEmotionRecognizer(
            args.model_path,
            'label_encoder_classes.npy'
        )
        recognizer.start_real_time_recognition()
