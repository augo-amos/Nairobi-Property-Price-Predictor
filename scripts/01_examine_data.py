"""
Step 1: Examine and understand your raw data structure
"""

import pandas as pd
import numpy as np

# Load your raw data
df = pd.read_csv('data/raw/nairobi_properties.csv')

print("="*60)
print("RAW DATA EXAMINATION")
print("="*60)

print(f"\n Dataset shape: {df.shape}")
print(f"\n Columns:")
for col in df.columns:
    print(f"  - {col}")

print(f"\n First 5 rows:")
print(df.head())

print(f"\n Data types:")
print(df.dtypes)

print(f"\n Missing values:")
print(df.isnull().sum())

print(f"\n  Sources distribution:")
print(df['source'].value_counts())

print(f"\n Listing type distribution:")
print(df['listing_type'].value_counts())

# Check for empty rows (like the ones with all nulls)
empty_rows = df[df['title'].isna() & df['price'].isna() & df['location'].isna()]
print(f"\n  Completely empty rows: {len(empty_rows)}")

# Save a sample of problematic rows for reference
problematic = df[df['title'].isna() | df['price'].isna()].head(20)
problematic.to_csv('data/raw/problematic_rows_sample.csv', index=False)
print("\n Sample of problematic rows saved to data/raw/problematic_rows_sample.csv")