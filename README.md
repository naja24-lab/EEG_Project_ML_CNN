EEG Classification Using Traditional Machine Learning

This project classifies EEG signals recorded before and during a mental arithmetic task using traditional machine learning algorithms and hand-crafted features.

Dataset:

Source: PhysioNet — EEG During Mental Arithmetic Tasks
Subjects: 36
Conditions per subject:
0 → Before task (baseline)
1 → During task (mental arithmetic)

Total samples: 72
Channels used: 19 (after removing reference and ECG channels)
Sampling rate (after preprocessing): 128 Hz
Features per sample: 320

Preprocessing & Feature Extraction:

EDF files read using MNE
Z-score normalization
Downsampling from 500 Hz → 128 Hz
Discrete Wavelet Transform (DWT) with db5 wavelet

4-level decomposition → bands:
D1: 32–64 Hz
D2: 16–32 Hz
D3: 8–16 Hz
D4: 4–8 Hz
A4: 0–4 Hz

16 hand-crafted features extracted for each channel × band:

Energy
Differential entropy
Mean
Interquartile range
Skewness
Kurtosis
4th, 5th, 6th moments
Weighted log energy
Teager energy
Median absolute deviation
Hurst exponent
Hjorth activity
Hjorth mobility
Hjorth complexity
Final feature matrix shape: 72 × 320

Machine Learning & Cross-Validation:

We evaluated four ML models with 10-fold cross validation:

Model	Mean Accuracy ± Std
SVM (RBF)	0.9589 ± 0.0629
KNN	0.9607 ± 0.0821
Random Forest	0.9179 ± 0.1084
AdaBoost	0.9018 ± 0.1431

SVM show the best performance (~96%).
