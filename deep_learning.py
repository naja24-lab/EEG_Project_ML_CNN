import numpy as np
import pywt
import mne
from scipy.interpolate import griddata
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import os
import glob
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "C:/Users/hp/OneDrive/Desktop/EEG_Project/data"
OUTPUT_DIR = "C:/Users/hp/OneDrive/Desktop/EEG_Project/processed_v2"
FEATURE_NAME = 'energy'

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("EEG CLASSIFICATION - FAST VERSION WITH PROGRESS")
print("="*70)
print(f"Feature: {FEATURE_NAME}")
print("="*70)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def compute_energy(signal):
    return np.sum(signal ** 2)

def compute_weighted_log_energy(signal):
    return np.sum(np.log(signal ** 2 + 1e-10))

def extract_feature(signal, feature_name):
    if feature_name == 'energy':
        return compute_energy(signal)
    elif feature_name == 'weighted_log_energy':
        return compute_weighted_log_energy(signal)
    return np.mean(signal)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_edf_files(data_dir):
    edf_files = glob.glob(os.path.join(data_dir, "*.edf"))
    
    if len(edf_files) == 0:
        print(f"ERROR: No .edf files found in {data_dir}")
        return []
    
    print(f"\n[1/5] Found {len(edf_files)} EDF files")
    
    data_list = []
    for edf_file in sorted(edf_files):
        try:
            basename = os.path.basename(edf_file)
            if '_' in basename:
                parts = basename.replace('.edf', '').split('_')
            else:
                parts = basename.replace('.edf', '').split()
            
            if len(parts) < 2:
                continue
                
            subject_str = parts[0].replace('Subject', '')
            condition_str = parts[1]
            subject_id = int(subject_str)
            condition = int(condition_str) - 1
            
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            channels_to_drop = ['EEG A1-A2', 'ECG ECG', 'A1-A2', 'ECG']
            existing_drops = [ch for ch in channels_to_drop if ch in raw.ch_names]
            if existing_drops:
                raw.drop_channels(existing_drops)
            
            data_list.append((subject_id, condition, raw))
        except Exception as e:
            print(f"Error loading {edf_file}: {e}")
    
    print(f"    Loaded {len(data_list)} recordings")
    return data_list

def preprocess_signal(raw, target_sfreq=128):
    data = raw.get_data()
    data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-10)
    
    if raw.info['sfreq'] != target_sfreq:
        raw_copy = raw.copy()
        raw_copy._data = data
        raw_copy.resample(target_sfreq, verbose=False)
        data = raw_copy.get_data()
    
    return data, raw.ch_names

def dwt_decomposition(signal, wavelet='db5', level=4):
    return pywt.wavedec(signal, wavelet, level=level)

def azimuthal_projection(electrode_positions):
    positions_2d = {}
    for ch_name, pos_3d in electrode_positions.items():
        x, y, z = pos_3d
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            positions_2d[ch_name] = (0, 0)
            continue
        x, y, z = x/r, y/r, z/r
        theta = np.arctan2(y, x)
        phi = np.arccos(np.clip(z, -1, 1))
        rho = phi / np.pi
        x_2d = rho * np.cos(theta)
        y_2d = rho * np.sin(theta)
        positions_2d[ch_name] = (x_2d, y_2d)
    return positions_2d

def generate_feature_map(feature_values, electrode_positions_2d, grid_size=200):
    points = []
    values = []
    
    for ch_name, value in feature_values.items():
        if ch_name in electrode_positions_2d:
            points.append(electrode_positions_2d[ch_name])
            values.append(value)
    
    if len(points) == 0:
        return np.zeros((grid_size, grid_size))
    
    points = np.array(points)
    values = np.array(values)
    
    x_min, x_max = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
    y_min, y_max = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1
    
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = griddata(points, values, (xi, yi), method='linear', fill_value=0)
    zi = np.nan_to_num(zi, 0)
    return zi

def create_feature_maps_for_recording(data, ch_names, positions_2d, feature_name):
    n_channels = data.shape[0]
    band_features = [dict() for _ in range(5)]
    
    for ch_idx in range(n_channels):
        signal = data[ch_idx, :]
        coeffs = dwt_decomposition(signal)
        
        for band_idx, coeff in enumerate(coeffs):
            try:
                feature_val = extract_feature(coeff, feature_name)
                band_features[band_idx][ch_names[ch_idx]] = feature_val
            except:
                band_features[band_idx][ch_names[ch_idx]] = 0
    
    stacked_maps = []
    for band_dict in band_features:
        feature_map = generate_feature_map(band_dict, positions_2d)
        stacked_maps.append(feature_map)
    
    return np.stack(stacked_maps, axis=-1)

def process_all_data(data_list, feature_name):
    print("\n[2/5] Setting up electrode positions...")
    montage = mne.channels.make_standard_montage('standard_1020')
    _, _, first_raw = data_list[0]
    ch_names = first_raw.ch_names
    
    pos_dict = {}
    for ch in ch_names:
        if ch in montage.ch_names:
            idx = montage.ch_names.index(ch)
            pos_dict[ch] = montage.dig[idx + 3]['r']
        else:
            ch_clean = ch.replace('EEG ', '').strip()
            if ch_clean in montage.ch_names:
                idx = montage.ch_names.index(ch_clean)
                pos_dict[ch] = montage.dig[idx + 3]['r']
    
    positions_2d = azimuthal_projection(pos_dict)
    print(f"    Configured {len(positions_2d)} electrodes")
    
    print(f"\n[3/5] Processing {len(data_list)} recordings...")
    X = []
    y = []
    
    for subject_id, condition, raw in tqdm(data_list):
        data, ch_names = preprocess_signal(raw)
        feature_maps = create_feature_maps_for_recording(data, ch_names, positions_2d, feature_name)
        X.append(feature_maps)
        y.append(condition)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[4/5] Normalizing feature maps...")
    for band_idx in range(X.shape[-1]):
        band_data = X[:, :, :, band_idx]
        mean = band_data.mean()
        std = band_data.std()
        if std > 0:
            X[:, :, :, band_idx] = (band_data - mean) / std
        print(f"    Band {band_idx+1}: mean={mean:.2e}, std={std:.2e}")
    
    return X, y

# ============================================================================
# CNN ARCHITECTURES - OPTIMIZED
# ============================================================================

def build_simple_cnn(input_shape=(200, 200, 5)):
    """Simple CNN - Fast and effective"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    return model

def build_aspp_cnn(input_shape=(200, 200, 5)):
    """Simplified ASPP architecture - faster than full version"""
    
    inputs = keras.Input(shape=input_shape)
    
    # Multi-scale feature extraction with parallel dilated convolutions
    conv1 = layers.Conv2D(32, 3, dilation_rate=1, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(32, 3, dilation_rate=2, padding='same', activation='relu')(inputs)
    conv3 = layers.Conv2D(32, 3, dilation_rate=4, padding='same', activation='relu')(inputs)
    
    # Concatenate multi-scale features
    concat = layers.Concatenate()([conv1, conv2, conv3])
    concat = layers.BatchNormalization()(concat)
    concat = layers.MaxPooling2D((2, 2))(concat)
    
    # Additional processing
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(concat)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global pooling
    x = layers.GlobalMaxPooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# ============================================================================
# TRAINING WITH PROGRESS
# ============================================================================

class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, fold_num):
        super().__init__()
        self.fold_num = fold_num
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        val_acc = logs.get('val_accuracy', 0)
        print(f"\r    Fold {self.fold_num} - Epoch {epoch+1:3d}: "
              f"val_acc={val_acc:.4f}, time={elapsed:.1f}s", end='', flush=True)

def train_with_kfold(X, y, model_builder, architecture_name, n_splits=10, batch_size=16, epochs=100):
    print(f"\n[5/5] Training {architecture_name}...")
    print(f"    Using {n_splits}-fold cross-validation")
    print(f"    Max epochs: {epochs}, Batch size: {batch_size}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_num = fold + 1
        print(f"\n\n  Fold {fold_num}/{n_splits} - Training...", flush=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        keras.backend.clear_session()
        
        model = model_builder()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=12,
            min_lr=1e-7,
            verbose=0
        )
        
        progress = ProgressCallback(fold_num)
        
        # Train with progress
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr, progress],
            verbose=0
        )
        
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        fold_accuracies.append(test_acc * 100)
        
        print(f"\n    → Test Accuracy: {test_acc * 100:.2f}%")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n    {'='*60}")
    print(f"    {architecture_name}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"    {'='*60}")
    
    return mean_acc, std_acc

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load data
    data_list = load_all_edf_files(DATA_DIR)
    
    if len(data_list) == 0:
        print("\nERROR: No data loaded!")
        return
    
    # Check for cached data
    X_file = os.path.join(OUTPUT_DIR, f'X_{FEATURE_NAME}_normalized.npy')
    y_file = os.path.join(OUTPUT_DIR, f'y_{FEATURE_NAME}_normalized.npy')
    
    if os.path.exists(X_file) and os.path.exists(y_file):
        print("\n[2-4/5] Loading cached normalized data...")
        X = np.load(X_file)
        y = np.load(y_file)
        print(f"    Loaded X: {X.shape}, y: {y.shape}")
    else:
        X, y = process_all_data(data_list, FEATURE_NAME)
        np.save(X_file, X)
        np.save(y_file, y)
        print(f"\n    Saved to: {OUTPUT_DIR}")
    
    print(f"\nData shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train both architectures
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Train Simple CNN first (faster)
    print("\n>>> Training Simple CNN first (Architecture 2)...")
    results2 = train_with_kfold(X, y, build_simple_cnn, "Simple CNN", epochs=100)
    
    # Ask user if they want to train the complex model
    print("\n>>> Train ASPP model? (slower, ~3-5 min per fold)")
    print("    Press Enter to train, or Ctrl+C to skip...")
    try:
        input()
        results1 = train_with_kfold(X, y, build_aspp_cnn, "ASPP CNN", epochs=100)
    except KeyboardInterrupt:
        print("\nSkipped ASPP training.")
        results1 = (0, 0)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    if results1[0] > 0:
        print(f"Architecture 1 (ASPP):     {results1[0]:.2f}% ± {results1[1]:.2f}%")
    print(f"Architecture 2 (Simple):   {results2[0]:.2f}% ± {results2[1]:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()