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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update these paths according to your setup
DATA_DIR = "C:/Users/hp/OneDrive/Desktop/EEG_Project/data"  # Where your .edf files are
OUTPUT_DIR = "C:/Users/hp/OneDrive/Desktop/EEG_Project/processed"  # Where to save processed data
FEATURE_NAME = 'energy'  # Options: 'energy', 'weighted_log_energy', 'hjorth_activity', etc.

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_all_edf_files(data_dir):
    """
    Load all EDF files from directory
    
    Returns:
    --------
    data_list : list of tuples
        Each tuple contains (subject_id, condition, raw_data)
        condition: 0 = before task (file ending in 1.edf)
                  1 = during task (file ending in 2.edf)
    """
    # Find all EDF files matching the pattern SubjectXX_Y.edf
    edf_files = glob.glob(os.path.join(data_dir, "*.edf"))
    
    if len(edf_files) == 0:
        print(f"No .edf files found in {data_dir}")
        print("Please check the DATA_DIR path at the top of the script")
        return []
    
    print(f"Found {len(edf_files)} EDF files")
    
    data_list = []
    
    for edf_file in sorted(edf_files):
        try:
            # Extract subject ID and condition from filename
            basename = os.path.basename(edf_file)
            
            # Parse filename (e.g., "Subject00_1.edf" or "Subject00 1.edf")
            if '_' in basename:
                parts = basename.replace('.edf', '').split('_')
            else:
                parts = basename.replace('.edf', '').split()
            
            if len(parts) < 2:
                print(f"Skipping {basename} - unexpected filename format")
                continue
                
            subject_str = parts[0].replace('Subject', '')
            condition_str = parts[1]
            
            subject_id = int(subject_str)
            condition = int(condition_str) - 1  # Convert 1,2 to 0,1
            
            # Load EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Drop reference and ECG channels if they exist
            channels_to_drop = ['EEG A1-A2', 'ECG ECG', 'A1-A2', 'ECG']
            existing_drops = [ch for ch in channels_to_drop if ch in raw.ch_names]
            if existing_drops:
                raw.drop_channels(existing_drops)
            
            data_list.append((subject_id, condition, raw))
            
        except Exception as e:
            print(f"Error loading {edf_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(data_list)} recordings")
    return data_list

def preprocess_signal(raw, target_sfreq=128):
    """
    Preprocess EEG signal: Z-score normalization and downsampling
    """
    # Get data
    data = raw.get_data()
    
    # Z-score normalization per channel
    data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-10)
    
    # Downsample
    if raw.info['sfreq'] != target_sfreq:
        raw_copy = raw.copy()
        raw_copy._data = data
        raw_copy.resample(target_sfreq, verbose=False)
        data = raw_copy.get_data()
    
    return data, raw.ch_names

# ============================================================================
# PART 2: WAVELET DECOMPOSITION
# ============================================================================

def dwt_decomposition(signal, wavelet='db5', level=4):
    """
    Perform Discrete Wavelet Transform decomposition
    
    Returns:
    --------
    coeffs : list
        List of wavelet coefficients [cA4, cD4, cD3, cD2, cD1]
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

# ============================================================================
# PART 3: FEATURE EXTRACTION
# ============================================================================

def compute_energy(signal):
    """Energy feature"""
    return np.sum(signal ** 2)

def compute_weighted_log_energy(signal):
    """Weighted log energy feature"""
    return np.sum(np.log(signal ** 2 + 1e-10))

def compute_differential_entropy(signal):
    """Differential entropy"""
    var = np.var(signal)
    if var <= 0:
        return 0
    return 0.5 * np.log(2 * np.pi * np.e * var)

def compute_hjorth_activity(signal):
    """Hjorth activity parameter"""
    return np.var(signal)

def compute_hjorth_mobility(signal):
    """Hjorth mobility parameter"""
    if len(signal) < 2:
        return 0
    diff_signal = np.diff(signal)
    var_signal = np.var(signal)
    if var_signal == 0:
        return 0
    return np.sqrt(np.var(diff_signal) / var_signal)

def compute_hjorth_complexity(signal):
    """Hjorth complexity parameter"""
    if len(signal) < 3:
        return 1
    diff_signal = np.diff(signal)
    diff_diff_signal = np.diff(diff_signal)
    
    var_signal = np.var(signal)
    var_diff = np.var(diff_signal)
    var_diff_diff = np.var(diff_diff_signal)
    
    if var_signal == 0 or var_diff == 0:
        return 1
    
    mobility = np.sqrt(var_diff / var_signal)
    mobility_diff = np.sqrt(var_diff_diff / var_diff)
    
    if mobility == 0:
        return 1
    
    return mobility_diff / mobility

def compute_hurst_exponent(signal):
    """Hurst exponent using R/S analysis"""
    n = len(signal)
    if n < 20:
        return 0.5
    
    mean_signal = np.mean(signal)
    deviation = signal - mean_signal
    cumulative_deviation = np.cumsum(deviation)
    
    R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
    S = np.std(signal)
    
    if S == 0 or R == 0:
        return 0.5
    
    return np.log(R / S) / np.log(n)

def compute_teager_energy(signal):
    """Mean Teager energy"""
    if len(signal) < 3:
        return 0
    teager = signal[1:-1]**2 - signal[:-2] * signal[2:]
    return np.mean(np.abs(teager))

def extract_feature(signal, feature_name):
    """
    Extract a specific feature from signal
    """
    feature_map = {
        'mean': lambda s: np.mean(s),
        'skewness': lambda s: float(np.mean(((s - np.mean(s)) / (np.std(s) + 1e-10))**3)),
        'kurtosis': lambda s: float(np.mean(((s - np.mean(s)) / (np.std(s) + 1e-10))**4)),
        'fourth_moment': lambda s: float(np.mean((s - np.mean(s))**4)),
        'fifth_moment': lambda s: float(np.mean((s - np.mean(s))**5)),
        'sixth_moment': lambda s: float(np.mean((s - np.mean(s))**6)),
        'iqr': lambda s: np.percentile(s, 75) - np.percentile(s, 25),
        'mad': lambda s: np.median(np.abs(s - np.median(s))),
        'energy': compute_energy,
        'weighted_log_energy': compute_weighted_log_energy,
        'diff_entropy': compute_differential_entropy,
        'hurst_exponent': compute_hurst_exponent,
        'hjorth_activity': compute_hjorth_activity,
        'hjorth_mobility': compute_hjorth_mobility,
        'hjorth_complexity': compute_hjorth_complexity,
        'teager_energy': compute_teager_energy
    }
    
    return feature_map[feature_name](signal)

# ============================================================================
# PART 4: 2D FEATURE MAP GENERATION
# ============================================================================

def azimuthal_projection(electrode_positions):
    """
    Convert 3D electrode positions to 2D using azimuthal equidistant projection
    """
    positions_2d = {}
    
    for ch_name, pos_3d in electrode_positions.items():
        x, y, z = pos_3d
        
        # Normalize to unit sphere
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            positions_2d[ch_name] = (0, 0)
            continue
        
        x, y, z = x/r, y/r, z/r
        
        # Azimuthal projection
        theta = np.arctan2(y, x)
        phi = np.arccos(np.clip(z, -1, 1))
        
        # Project to 2D
        rho = phi / np.pi
        x_2d = rho * np.cos(theta)
        y_2d = rho * np.sin(theta)
        
        positions_2d[ch_name] = (x_2d, y_2d)
    
    return positions_2d

def generate_feature_map(feature_values, electrode_positions_2d, grid_size=200):
    """
    Generate 2D feature map using Delaunay triangulation interpolation
    """
    # Extract positions and values
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
    
    # Create grid
    x_min, x_max = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
    y_min, y_max = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1
    
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate
    zi = griddata(points, values, (xi, yi), method='linear', fill_value=0)
    zi = np.nan_to_num(zi, 0)
    
    return zi

def create_feature_maps_for_recording(data, ch_names, positions_2d, feature_name='energy'):
    """
    Create stacked 2D feature maps for all 5 frequency bands
    
    Returns:
    --------
    stacked_maps : numpy array
        Shape (200, 200, 5) - 5 frequency bands stacked
    """
    n_channels = data.shape[0]
    
    # Extract features for each band
    band_features = [dict() for _ in range(5)]
    
    for ch_idx in range(n_channels):
        signal = data[ch_idx, :]
        coeffs = dwt_decomposition(signal)
        
        # Process each band (A4, D4, D3, D2, D1)
        for band_idx, coeff in enumerate(coeffs):
            try:
                feature_val = extract_feature(coeff, feature_name)
                band_features[band_idx][ch_names[ch_idx]] = feature_val
            except:
                band_features[band_idx][ch_names[ch_idx]] = 0
    
    # Generate 2D maps for each band
    stacked_maps = []
    for band_dict in band_features:
        feature_map = generate_feature_map(band_dict, positions_2d)
        stacked_maps.append(feature_map)
    
    stacked_maps = np.stack(stacked_maps, axis=-1)  # Shape: (200, 200, 5)
    
    return stacked_maps

# ============================================================================
# PART 5: COMPLETE DATA PROCESSING PIPELINE
# ============================================================================

def process_all_data(data_dir, feature_name='energy', output_dir='processed'):
    """
    Process all EDF files and create feature maps
    """
    print("=" * 70)
    print("EEG DATA PROCESSING PIPELINE")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Feature: {feature_name}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Load all EDF files
    print("\n[1/4] Loading EDF files...")
    data_list = load_all_edf_files(data_dir)
    
    if len(data_list) == 0:
        print("\nERROR: No data loaded. Please check:")
        print(f"  1. DATA_DIR path is correct: {data_dir}")
        print("  2. EDF files exist in that directory")
        print("  3. Filenames match pattern: SubjectXX_Y.edf or SubjectXX Y.edf")
        return None, None
    
    # Get electrode positions from standard montage
    print("\n[2/4] Setting up electrode positions...")
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Get first recording to determine channel names
    _, _, first_raw = data_list[0]
    ch_names = first_raw.ch_names
    
    # Create position dictionary
    pos_dict = {}
    for ch in ch_names:
        # Try exact match first
        if ch in montage.ch_names:
            idx = montage.ch_names.index(ch)
            pos_dict[ch] = montage.dig[idx + 3]['r']
        else:
            # Try without 'EEG ' prefix
            ch_clean = ch.replace('EEG ', '').strip()
            if ch_clean in montage.ch_names:
                idx = montage.ch_names.index(ch_clean)
                pos_dict[ch] = montage.dig[idx + 3]['r']
    
    print(f"Found positions for {len(pos_dict)}/{len(ch_names)} channels")
    
    # Project to 2D
    positions_2d = azimuthal_projection(pos_dict)
    
    # Process each recording
    print(f"\n[3/4] Processing {len(data_list)} recordings...")
    X = []
    y = []
    
    for subject_id, condition, raw in tqdm(data_list):
        # Preprocess
        data, ch_names = preprocess_signal(raw)
        
        # Create feature maps
        feature_maps = create_feature_maps_for_recording(
            data, ch_names, positions_2d, feature_name
        )
        
        X.append(feature_maps)
        y.append(condition)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[4/4] Saving processed data...")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Save
    output_path_X = os.path.join(output_dir, f'X_{feature_name}.npy')
    output_path_y = os.path.join(output_dir, f'y_{feature_name}.npy')
    
    np.save(output_path_X, X)
    np.save(output_path_y, y)
    
    print(f"\nSaved to:")
    print(f"  {output_path_X}")
    print(f"  {output_path_y}")
    print("\nProcessing complete!")
    
    return X, y

# ============================================================================
# PART 6: CNN ARCHITECTURES
# ============================================================================

def build_cnn_architecture_1(input_shape=(200, 200, 5)):
    """Build CNN Architecture 1 with ASPP and FPN modules"""
    
    def aspp_module(x, filters=64, name_prefix='aspp'):
        dilation_rates = [1, 2, 4, 6, 12]
        conv_outputs = []
        
        for rate in dilation_rates:
            conv = layers.Conv2D(
                filters, kernel_size=2, strides=1,
                dilation_rate=rate, padding='same',
                activation='relu', name=f'{name_prefix}_conv_{rate}'
            )(x)
            conv_outputs.append(conv)
        
        concat = layers.Concatenate(name=f'{name_prefix}_concat')(conv_outputs)
        output = layers.Conv2D(
            filters, kernel_size=1, strides=2,
            padding='same', activation='relu',
            name=f'{name_prefix}_1x1'
        )(concat)
        return output
    
    inputs = keras.Input(shape=input_shape)
    
    # Three ASPP modules
    aspp1 = aspp_module(inputs, name_prefix='aspp1')
    aspp2 = aspp_module(aspp1, name_prefix='aspp2')
    aspp3 = aspp_module(aspp2, name_prefix='aspp3')
    
    # FPN
    aspp1_1x1 = layers.Conv2D(64, 1, padding='same')(aspp1)
    aspp2_1x1 = layers.Conv2D(64, 1, padding='same')(aspp2)
    
    aspp3_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(aspp3)
    fpn1 = layers.Add()([aspp3_up, aspp2_1x1])
    
    fpn1_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(fpn1)
    fpn2 = layers.Add()([fpn1_up, aspp1_1x1])
    
    # Pooling
    fpn1_pool = layers.GlobalMaxPooling2D()(fpn1)
    fpn2_pool = layers.GlobalMaxPooling2D()(fpn2)
    aspp3_pool = layers.GlobalMaxPooling2D()(aspp3)
    
    concat = layers.Concatenate()([fpn1_pool, fpn2_pool, aspp3_pool])
    dense = layers.Dense(64, activation='relu')(concat)
    outputs = layers.Dense(2, activation='softmax')(dense)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='CNN_Arch1')

def build_cnn_architecture_2(input_shape=(200, 200, 5)):
    """Build CNN Architecture 2 (simpler)"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ], name='CNN_Arch2')
    return model

# ============================================================================
# PART 7: TRAINING
# ============================================================================

def train_with_kfold(X, y, model_builder, n_splits=10, batch_size=16, epochs=100):
    """Train model with K-fold cross-validation"""
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_builder()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, min_delta=0.001,
            restore_best_weights=True, verbose=0
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=0
        )
        
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)
        fold_accuracies.append(test_acc * 100)
        print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accuracies
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("EEG DEEP LEARNING CLASSIFICATION")
    print("=" * 70)
    
    # Check if processed data already exists
    X_file = os.path.join(OUTPUT_DIR, f'X_{FEATURE_NAME}.npy')
    y_file = os.path.join(OUTPUT_DIR, f'y_{FEATURE_NAME}.npy')
    
    if os.path.exists(X_file) and os.path.exists(y_file):
        print(f"\nLoading existing processed data for feature: {FEATURE_NAME}")
        X = np.load(X_file)
        y = np.load(y_file)
        print(f"Loaded X: {X.shape}, y: {y.shape}")
    else:
        print(f"\nProcessed data not found. Processing raw EDF files...")
        X, y = process_all_data(DATA_DIR, FEATURE_NAME, OUTPUT_DIR)
        
        if X is None:
            return
    
    # Train models
    print("\n" + "=" * 70)
    print(f"TRAINING MODELS - Feature: {FEATURE_NAME}")
    print("=" * 70)
    
    print("\n--- Architecture 1 (ASPP + FPN) ---")
    results_arch1 = train_with_kfold(X, y, build_cnn_architecture_1)
    print(f"\nResults: {results_arch1['mean_accuracy']:.2f}% ± {results_arch1['std_accuracy']:.2f}%")
    
    print("\n--- Architecture 2 (Simple CNN) ---")
    results_arch2 = train_with_kfold(X, y, build_cnn_architecture_2)
    print(f"\nResults: {results_arch2['mean_accuracy']:.2f}% ± {results_arch2['std_accuracy']:.2f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()