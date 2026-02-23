"""
Step 2: Clean and structure the scraped data
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

print("Loading raw data...")
df = pd.read_csv('data/raw/nairobi_properties.csv')
print(f"Initial rows: {len(df)}")

# 1. Remove completely empty rows
df = df.dropna(how='all')
print(f"After removing empty rows: {len(df)}")

# 2. Remove duplicate listings (based on URL if available, else title+price)
if 'url' in df.columns:
    # Remove rows with no URL but keep those with data
    df_with_url = df[df['url'].notna()].drop_duplicates(subset=['url'], keep='first')
    df_without_url = df[df['url'].isna()]
    df = pd.concat([df_with_url, df_without_url], ignore_index=True)
else:
    df = df.drop_duplicates(subset=['title', 'price'], keep='first')
print(f"After removing duplicates: {len(df)}")

# 3. Clean price column
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    # Convert to string and extract numbers
    price_str = str(price_str)
    # Find pattern like "KSh 160,000,000" or "160,000,000"
    match = re.search(r'(?:KSh\s*)?([\d,]+)', price_str)
    if match:
        # Remove commas and convert to int
        return int(match.group(1).replace(',', ''))
    return np.nan

df['price_clean'] = df['price'].apply(clean_price)

# 4. Extract location (remove "Nairobi" suffix/prefix)
def clean_location(loc_str):
    if pd.isna(loc_str):
        return np.nan
    loc_str = str(loc_str)
    # Remove "Nairobi" and clean up
    loc = loc_str.replace('Nairobi', '').strip(', ')
    # Take first part if multiple
    if ',' in loc:
        loc = loc.split(',')[0].strip()
    return loc if loc else np.nan

df['location_clean'] = df['location'].apply(clean_location)

# 5. Extract bedrooms
def extract_bedrooms(text):
    if pd.isna(text):
        return np.nan
    text = str(text)
    # Look for patterns like "5 bed", "5-bed", "5bed", "5 Bedroom"
    match = re.search(r'(\d+)\s*-?\s*[Bb]ed', text)
    if match:
        return int(match.group(1))
    return np.nan

df['bedrooms_clean'] = df['bedrooms'].apply(extract_bedrooms)
# Also check title for bedrooms if missing
mask = df['bedrooms_clean'].isna()
df.loc[mask, 'bedrooms_clean'] = df.loc[mask, 'title'].apply(extract_bedrooms)

# 6. Extract bathrooms
def extract_bathrooms(text):
    if pd.isna(text):
        return np.nan
    text = str(text)
    # Look for patterns like "3 bath", "3-bath", "3bath", "3 Bathroom"
    match = re.search(r'(\d+)\s*-?\s*[Bb]ath', text)
    if match:
        return int(match.group(1))
    return np.nan

df['bathrooms_clean'] = df['bathrooms'].apply(extract_bathrooms)
# Also check title
mask = df['bathrooms_clean'].isna()
df.loc[mask, 'bathrooms_clean'] = df.loc[mask, 'title'].apply(extract_bathrooms)

# 7. Extract size in sqm
def extract_size(text):
    if pd.isna(text):
        return np.nan
    text = str(text)
    # Look for patterns like "150 m²", "150m²", "150 sqm"
    match = re.search(r'(\d+)\s*(?:m²|sqm|m2)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return np.nan

df['size_clean'] = df['size_sqm'].apply(extract_size)
# Also check title
mask = df['size_clean'].isna()
df.loc[mask, 'size_clean'] = df.loc[mask, 'title'].apply(extract_size)

# 8. Extract property type
def extract_property_type(title, raw_type):
    if pd.notna(raw_type) and raw_type:
        return raw_type
    if pd.isna(title):
        return 'Unknown'
    title = str(title).lower()
    if 'apartment' in title:
        return 'Apartment'
    elif 'house' in title:
        return 'House'
    elif 'townhouse' in title or 'town house' in title:
        return 'Townhouse'
    elif 'villa' in title:
        return 'Villa'
    elif 'studio' in title:
        return 'Studio'
    else:
        return 'Unknown'

df['property_type_clean'] = df.apply(
    lambda row: extract_property_type(row['title'], row['property_type']), 
    axis=1
)

# 9. Filter valid listings (must have price and either location or title)
df = df[df['price_clean'].notna()]
df = df[df['location_clean'].notna() | df['title'].notna()]
print(f"After filtering invalid: {len(df)}")

# 10. Remove extreme outliers (prices that seem unrealistic)
# Keep prices between 500,000 and 500,000,000 KES
df = df[(df['price_clean'] >= 500000) & (df['price_clean'] <= 500000000)]
print(f"After price filtering: {len(df)}")

# 11. Create final clean dataset
clean_df = pd.DataFrame({
    'id': range(1, len(df) + 1),
    'source': df['source'],
    'listing_type': df['listing_type'],
    'title': df['title'],
    'price': df['price_clean'],
    'location': df['location_clean'],
    'bedrooms': df['bedrooms_clean'],
    'bathrooms': df['bathrooms_clean'],
    'size_sqm': df['size_clean'],
    'property_type': df['property_type_clean'],
    'url': df['url'],
    'scraped_at': df['scraped_at']
})

# Fill missing numeric values with median
for col in ['bedrooms', 'bathrooms', 'size_sqm']:
    median_val = clean_df[col].median()
    clean_df[col] = clean_df[col].fillna(median_val)

# Fill missing categorical
clean_df['property_type'] = clean_df['property_type'].fillna('Unknown')
clean_df['location'] = clean_df['location'].fillna('Unknown')

# 12. Create data dictionary
data_dict = pd.DataFrame({
    'Column': clean_df.columns.tolist(),
    'Description': [
        'Unique identifier',
        'Source website (PigiaMe/BuyRentKenya/Jiji/Property24)',
        'Sale or Rent',
        'Listing title',
        'Price in KES',
        'Neighborhood in Nairobi',
        'Number of bedrooms',
        'Number of bathrooms',
        'Size in square meters',
        'Type of property',
        'Original URL',
        'Date scraped'
    ],
    'Data Type': ['int', 'string', 'string', 'string', 'int', 'string', 
                  'float', 'float', 'float', 'string', 'string', 'string']
})

# Save cleaned data
clean_df.to_csv('data/processed/clean_listings.csv', index=False)
data_dict.to_csv('data/processed/data_dictionary.csv', index=False)

print("\n" + "="*60)
print("CLEANING COMPLETE")
print("="*60)
print(f"\n Final clean dataset: {len(clean_df)} rows")
print(f"\n Price statistics:")
print(clean_df['price'].describe().apply(lambda x: f'KES {x:,.0f}'))
print(f"\n Top locations:")
print(clean_df['location'].value_counts().head(10))
print(f"\n Property types:")
print(clean_df['property_type'].value_counts())
print(f"\n Bedrooms distribution:")
print(clean_df['bedrooms'].value_counts().sort_index())

# Save sample for verification
clean_df.head(20).to_csv('data/processed/sample_clean.csv', index=False)
print("\n Sample saved to data/processed/sample_clean.csv")