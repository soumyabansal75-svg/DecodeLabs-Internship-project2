import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)

# ==========================================
# STEP 1: LOAD AND UNDERSTAND DATA
# ==========================================
print("=" * 60)
print(" STEP 1: DATA LOADING & EXPLORATION")
print("=" * 60)

# Option 1: Load your own dataset
# Replace 'dataset.csv' with your actual file path
try:
    df = pd.read_csv('dataset.csv')
    print("✓ Dataset loaded successfully!")
except FileNotFoundError:
    print("⚠ File not found. Using generated sample data.")
    # Generate sample data for demonstration
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=200, 
        n_features=4, 
        n_classes=2, 
        n_informative=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=['Temperature', 'Pressure', 'Vibration', 'Speed'])
    df['Target'] = y

# Display dataset info
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 10 rows:\n{df.head(10)}")
print(f"\nColumn Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nTarget Distribution:\n{df['Target'].value_counts()}")

# ==========================================
# STEP 2: FEATURE & TARGET SEPARATION
# ==========================================
print("\n" + "=" * 60)
print(" STEP 2: FEATURE & TARGET SEPARATION")
print("=" * 60)

X = df.iloc[:, :-1]  # Features (all columns except last)
y = df.iloc[:, -1]   # Target (last column)

print(f"\nFeatures (X): {list(X.columns)}")
print(f"Features Shape: {X.shape}")
print(f"Target Shape:  {y.shape}")
print(f"Target Classes: {y.unique()}")

# ==========================================
# STEP 3: TRAIN-TEST SPLIT
# ==========================================
print("\n" + "=" * 60)
print(" STEP 3: TRAIN-TEST SPLIT (80-20)")
print("=" * 60)

# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True,      # Remove order bias
    stratify=y         # Maintain class distribution
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print(f"Ratio: {len(X_train)}:{len(X_test)}")
print(f"\nTraining Target Distribution:\n{y_train.value_counts().sort_index()}")
print(f"\nTesting Target Distribution:\n{y_test.value_counts().sort_index()}")

# ==========================================
# STEP 4: FEATURE SCALING
# ==========================================
print("\n" + "=" * 60)
print(" STEP 4: FEATURE SCALING (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()

# Fit on training data, transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nBefore Scaling:")
print(f"  Mean: {X_train.mean().round(2).values}")
print(f"  Std:  {X_train.std().round(2).values}")

print("\nAfter Scaling:")
print(f"  Mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"  Std:  {X_train_scaled.std(axis=0).round(4)}")

# ==========================================
# STEP 5: MODEL TRAINING
# ==========================================
print("\n" + "=" * 60)
print(" STEP 5: MODEL TRAINING (Logistic Regression)")
print("=" * 60)

# Initialize and train model
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train_scaled, y_train)

print(f"\nModel: Logistic Regression")
print(f"Training samples: {len(X_train)}")
print(f"Coefficients: {model.coef_[0].round(4)}")
print(f"Intercept:   {model.intercept_[0]:.4f}")

# ==========================================
# STEP 6: PREDICTIONS
# ==========================================
print("\n" + "=" * 60)
print(" STEP 6: MAKING PREDICTIONS")
print("=" * 60)

y_pred = model.predict(X_test_scaled)

print(f"\nPredicted values (first 10): {y_pred[:10]}")
print(f"Actual values (first 10):    {y_test.values[:10]}")

# ==========================================
# STEP 7: VALIDATION METRICS
# ==========================================
print("\n" + "=" * 60)
print(" STEP 7: MODEL EVALUATION")
print("=" * 60)

# Individual metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"                 Class 0  Class 1")
print(f"  Actual Class 0  {cm[0][0]:^7}  {cm[0][1]:^7}")
print(f"  Actual Class 1  {cm[1][0]:^7}  {cm[1][1]:^7}")

print(f"\n📈 PERFORMANCE METRICS:")
print(f"  ✓ Accuracy:  {accuracy:.2%}")
print(f"  ✓ Precision: {precision:.2%}")
print(f"  ✓ Recall:    {recall:.2%}")
print(f"  ✓ F1 Score:  {f1:.2%}")

print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# STEP 8: SAMPLE PREDICTIONS
# ==========================================
print("\n" + "=" * 60)
print(" STEP 8: NEW SAMPLE PREDICTIONS")
print("=" * 60)

# Test with new data samples
new_samples = np.array([
    [0.5, 0.3, 0.8, 0.2],   # Sample 1
    [-0.8, 0.5, -0.3, 0.9], # Sample 2
])

new_scaled = scaler.transform(new_samples)
new_predictions = model.predict(new_scaled)

print(f"\nNew Sample 1: {new_samples[0]} → Predicted Class: {new_predictions[0]}")
print(f"New Sample 2: {new_samples[1]} → Predicted Class: {new_predictions[1]}")

print("\n" + "=" * 60)
print(" ✅ PROJECT 2 COMPLETE")
print("=" * 60)
