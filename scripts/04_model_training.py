"""
Step 4: Flexible Model Training - Uses whatever data is available
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed/features_listings.csv')
print(f"Loaded {len(df)} records")
print("\nMissing values by column:")
print(df.isnull().sum())

# 1. First, let's see what data we actually have
print("\n" + "="*60)
print("DATA AVAILABILITY ANALYSIS")
print("="*60)

# Check each important column
important_cols = ['price', 'bedrooms', 'bathrooms', 'size_sqm', 'location', 'property_type']
for col in important_cols:
    if col in df.columns:
        available = df[col].notna().sum()
        pct = (available / len(df)) * 100
        print(f"{col}: {available}/{len(df)} records ({pct:.1f}%)")

# 2. Create different model configurations based on available data
print("\n" + "="*60)
print("TRAINING MULTIPLE MODELS WITH DIFFERENT FEATURES")
print("="*60)

models_to_try = []

# Model 1: Using only location and price (simplest)
if 'location' in df.columns and df['price'].notna().any():
    models_to_try.append({
        'name': 'Location Only',
        'features': ['location'],
        'categorical': ['location']
    })

# Model 2: Using bedrooms and location
if all(col in df.columns for col in ['bedrooms', 'location']):
    if df['bedrooms'].notna().any() and df['location'].notna().any():
        models_to_try.append({
            'name': 'Bedrooms + Location',
            'features': ['bedrooms', 'location'],
            'categorical': ['location']
        })

# Model 3: Using bedrooms, bathrooms, and location
if all(col in df.columns for col in ['bedrooms', 'bathrooms', 'location']):
    models_to_try.append({
        'name': 'Bedrooms + Bathrooms + Location',
        'features': ['bedrooms', 'bathrooms', 'location'],
        'categorical': ['location']
    })

# Model 4: Using size and location
if all(col in df.columns for col in ['size_sqm', 'location']):
    if df['size_sqm'].notna().any():
        models_to_try.append({
            'name': 'Size + Location',
            'features': ['size_sqm', 'location'],
            'categorical': ['location']
        })

# Model 5: Using all available features (but only where they exist)
all_features = []
for col in ['bedrooms', 'bathrooms', 'size_sqm', 'location', 'property_type']:
    if col in df.columns and df[col].notna().any():
        all_features.append(col)

if len(all_features) >= 2:  # At least 2 features
    models_to_try.append({
        'name': 'All Available',
        'features': all_features,
        'categorical': [f for f in all_features if f in ['location', 'property_type']]
    })

print(f"\nWill try {len(models_to_try)} different model configurations")

# 3. Train and evaluate each model
results = {}
best_model = None
best_mae = float('inf')
best_model_name = None
best_features = None

for model_config in models_to_try:
    print(f"\n" + "-"*40)
    print(f"Training: {model_config['name']}")
    print("-"*40)
    
    # Prepare data for this configuration
    feature_cols = model_config['features']
    categorical_cols = model_config.get('categorical', [])
    
    # Create a working dataframe
    working_df = df[feature_cols + ['price']].copy()
    
    # Drop rows where price is missing
    working_df = working_df.dropna(subset=['price'])
    
    # Handle missing values in features
    for col in feature_cols:
        if col in ['bedrooms', 'bathrooms', 'size_sqm']:
            # For numeric columns, fill with median
            median_val = working_df[col].median()
            working_df[col] = working_df[col].fillna(median_val)
        elif col in categorical_cols:
            # For categorical, fill with 'Unknown'
            working_df[col] = working_df[col].fillna('Unknown')
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        working_df[f'{col}_encoded'] = le.fit_transform(working_df[col].astype(str))
        encoders[col] = le
        # Replace original with encoded
        feature_cols.remove(col)
        feature_cols.append(f'{col}_encoded')
    
    print(f"Records available: {len(working_df)}")
    
    if len(working_df) < 10:
        print(" Not enough records, skipping...")
        continue
    
    # Prepare X and y
    X = working_df[feature_cols]
    y = working_df['price']
    
    # Log transform target
    y_log = np.log(y)
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )
    except ValueError as e:
        print(f" Error splitting data: {e}")
        continue
    
    # Train models
    for model_name, model in [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
    ]:
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_log = model.predict(X_test)
            y_pred = np.exp(y_pred_log)
            y_actual = np.exp(y_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            model_key = f"{model_config['name']} - {model_name}"
            results[model_key] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'model': model,
                'features': feature_cols,
                'encoders': encoders if categorical_cols else None,
                'n_samples': len(working_df)
            }
            
            print(f"{model_name}: MAE = KES {mae:,.0f}, R² = {r2:.3f}")
            
            # Track best model
            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_model_name = model_key
                best_features = feature_cols
                best_encoders = encoders if categorical_cols else None
                
        except Exception as e:
            print(f" Error training {model_name}: {e}")

# 4. Display all results
print("\n" + "="*60)
print("ALL MODEL RESULTS")
print("="*60)

if results:
    results_df = pd.DataFrame({
        name: {
            'MAE (KES)': f"{metrics['MAE']:,.0f}",
            'R²': f"{metrics['R2']:.3f}",
            'Samples': metrics['n_samples']
        }
        for name, metrics in results.items()
    }).T
    
    print(results_df)
    results_df.to_csv('data/processed/all_model_results.csv')
    
    # 5. Save best model
    print("\n" + "="*60)
    print("BEST MODEL")
    print("="*60)
    print(f" {best_model_name}")
    print(f"   MAE: KES {best_mae:,.0f}")
    print(f"   Features: {best_features}")
    print(f"   Training samples: {results[best_model_name]['n_samples']}")
    
    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(best_features, 'models/best_features.pkl')
    if best_encoders:
        joblib.dump(best_encoders, 'models/best_encoders.pkl')
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'mae': best_mae,
        'rmse': results[best_model_name]['RMSE'],
        'r2': results[best_model_name]['R2'],
        'n_samples': results[best_model_name]['n_samples'],
        'features': str(best_features)
    }
    pd.Series(metadata).to_csv('models/best_model_metadata.csv')
    
    print("\n Best model saved to models/best_model.pkl")
    
else:
    print(" No models could be trained successfully!")