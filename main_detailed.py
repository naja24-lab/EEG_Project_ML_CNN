import os
import mne
import numpy as np
import pywt
from scipy import stats, signal
from scipy.stats import kurtosis, skew, iqr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_and_preprocess_edf(file_path, target_freq=128):
    """Load and preprocess EDF file."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    channels_to_drop = ['EEG A1-A2', 'ECG ECG']
    channels_to_drop = [ch for ch in channels_to_drop if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    raw = raw.resample(target_freq, npad='auto', verbose=False)
    return raw.get_data()

def calculate_handcrafted_features(data):
    """Calculate all 16 handcrafted features."""
    features = []
    
    # Time-domain features
    features.append(np.mean(data, axis=1))  # Mean
    features.append(np.var(data, axis=1))   # Variance
    features.append(skew(data, axis=1))     # Skewness
    features.append(kurtosis(data, axis=1)) # Kurtosis
    features.append(iqr(data, axis=1))      # IQR
    
    # Higher moments
    features.append(stats.moment(data, moment=4, axis=1))
    features.append(stats.moment(data, moment=5, axis=1))
    features.append(stats.moment(data, moment=6, axis=1))
    
    # Energy features
    features.append(np.sum(data**2, axis=1))  # Energy
    features.append(np.log1p(np.sum(data**2, axis=1)))  # Log energy
    features.append(np.sum(np.diff(data)**2, axis=1))   # Teager energy
    
    # Hjorth parameters
    def hjorth_parameters(x):
        x = np.array(x)
        x_diff = np.diff(x)
        activity = np.var(x)
        mobility = np.sqrt(np.var(x_diff) / activity)
        complexity = np.sqrt(np.var(np.diff(x_diff)) / np.var(x_diff)) / mobility
        return activity, mobility, complexity
    
    hjorth_feats = np.array([hjorth_parameters(ch) for ch in data])
    features.extend([hjorth_feats[:,0], hjorth_feats[:,1], hjorth_feats[:,2]])
    
    # Hurst exponent
    def hurst(x):
        x = np.cumsum(x - np.mean(x))
        r = np.max(x) - np.min(x)
        s = np.std(x)
        return np.log(r/s) / np.log(len(x))
    
    features.append(np.array([hurst(ch) for ch in data]))
    
    # Differential entropy
    features.append(0.5 * np.log(2 * np.pi * np.e * np.var(data, axis=1)))
    
    return np.column_stack(features)

def process_files(data_dir):
    """Process all EDF files and extract features."""
    X, y = [], []
    subjects = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf')])
    
    for filename in tqdm(subjects, desc="Processing files"):
        file_path = os.path.join(data_dir, filename)
        try:
            # Load and preprocess
            data = load_and_preprocess_edf(file_path)
            
            # Extract features
            features = calculate_handcrafted_features(data)
            X.append(features.flatten())  # Flatten all features
            
            # Get label (0 for rest, 1 for task)
            label = 1 if filename.endswith('_2.edf') else 0
            y.append(label)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return np.array(X), np.array(y)

def main():
    # Set your data directory
    data_dir = r'c:\Users\hp\OneDrive\Desktop\EEG_Project\data'
    
    print("Processing data and extracting features...")
    X, y = process_files(data_dir)
    
    print("\nDataset statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define classifiers
    classifiers = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    # Store results
    results = {name: [] for name in classifiers.keys()}
    
    # 10-Fold CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=80)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            print(f"Fold {fold} - {name}: {acc:.4f}")
    
    # Print final results
    print("\nMean accuracy ± std across 10 folds:")
    for name in classifiers.keys():
        mean_acc = np.mean(results[name])
        std_acc = np.std(results[name])
        print(f"{name}: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    main()