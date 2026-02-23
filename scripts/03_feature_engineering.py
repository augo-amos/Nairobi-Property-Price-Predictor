"""
Step 3: Feature engineering and exploratory analysis (FIXED VERSION)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load cleaned data
df = pd.read_csv('data/processed/clean_listings.csv')
print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# 1. Create new features
print("\n Creating new features...")

# First, ensure numeric columns are properly converted
numeric_cols = ['price', 'bedrooms', 'bathrooms', 'size_sqm']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Price per sqm (if size available)
df['price_per_sqm'] = df['price'] / df['size_sqm']
df['price_per_sqm'] = df['price_per_sqm'].replace([np.inf, -np.inf], np.nan)

# Price per bedroom
df['price_per_bedroom'] = df['price'] / df['bedrooms']
df['price_per_bedroom'] = df['price_per_bedroom'].replace([np.inf, -np.inf], np.nan)

# Log price for modeling
df['log_price'] = np.log(df['price'])

# Size category
def size_category(size):
    if pd.isna(size):
        return 'Unknown'
    if size < 70:
        return 'Small (<70 sqm)'
    elif size < 120:
        return 'Medium (70-120 sqm)'
    elif size < 200:
        return 'Large (120-200 sqm)'
    else:
        return 'Very Large (>200 sqm)'

df['size_category'] = df['size_sqm'].apply(size_category)

# Price category
def price_category(price):
    if price < 5_000_000:
        return 'Budget (<5M)'
    elif price < 15_000_000:
        return 'Mid-range (5-15M)'
    elif price < 30_000_000:
        return 'Premium (15-30M)'
    else:
        return 'Luxury (>30M)'

df['price_category'] = df['price'].apply(price_category)

# 2. Basic statistics by group
print("\n Average price by location (top 15):")
# Filter out unknown locations and ensure we have enough data
valid_df = df[df['location'] != 'Unknown']
loc_stats = valid_df.groupby('location')['price'].agg(['mean', 'median', 'count']).sort_values('median', ascending=False)
loc_stats = loc_stats[loc_stats['count'] >= 2]  # Only locations with at least 2 listings
print(loc_stats.head(15).round(0).map(lambda x: f'KES {x:,.0f}' if pd.notna(x) else x))

print("\n Average price by property type:")
type_stats = df.groupby('property_type')['price'].agg(['mean', 'median', 'count']).sort_values('median', ascending=False)
print(type_stats.round(0).map(lambda x: f'KES {x:,.0f}' if pd.notna(x) else x))

# 3. Correlation analysis - only on complete cases
print("\n Correlation matrix (only on complete cases):")
# Select only numeric columns and drop rows with missing values
corr_cols = ['price', 'bedrooms', 'bathrooms', 'size_sqm', 'price_per_sqm']
corr_df = df[corr_cols].dropna()
print(f"Using {len(corr_df)} complete cases for correlation")
if len(corr_df) > 0:
    corr_matrix = corr_df.corr()
    print(corr_matrix.round(3))
else:
    print("Not enough complete cases for correlation")

# 4. Create visualizations
print("\n Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Price distribution
axes[0, 0].hist(df['price'].dropna()/1e6, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Price (Millions KES)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Price Distribution')
if len(df['price'].dropna()) > 0:
    axes[0, 0].axvline(df['price'].median()/1e6, color='red', linestyle='--', 
                      label=f"Median: {df['price'].median()/1e6:.1f}M")
    axes[0, 0].legend()

# Log price distribution
log_prices = df['log_price'].dropna()
if len(log_prices) > 0:
    axes[0, 1].hist(log_prices, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Log Price')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Log Price Distribution')

# Price by location (top 10 by count)
top_locations = df['location'].value_counts().head(10).index
df_top = df[df['location'].isin(top_locations) & (df['location'] != 'Unknown')]
if len(df_top) > 0:
    sns.boxplot(data=df_top, x='location', y='price', ax=axes[0, 2])
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=45, ha='right')
    axes[0, 2].set_title('Price Distribution by Location')
    axes[0, 2].set_ylabel('Price (KES)')

# Price vs Size (only where both exist)
size_price_df = df[['size_sqm', 'price']].dropna()
if len(size_price_df) > 0:
    axes[1, 0].scatter(size_price_df['size_sqm'], size_price_df['price']/1e6, alpha=0.5, s=30)
    axes[1, 0].set_xlabel('Size (sqm)')
    axes[1, 0].set_ylabel('Price (Millions KES)')
    axes[1, 0].set_title('Price vs Size')

# Price by Bedrooms
bedroom_df = df[['bedrooms', 'price']].dropna()
if len(bedroom_df) > 0:
    sns.boxplot(data=bedroom_df, x='bedrooms', y='price', ax=axes[1, 1])
    axes[1, 1].set_title('Price by Number of Bedrooms')
    axes[1, 1].set_ylabel('Price (KES)')

# Property type distribution
type_counts = df['property_type'].value_counts()
if len(type_counts) > 0:
    axes[1, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[1, 2].set_title('Property Type Distribution')

plt.tight_layout()
plt.savefig('data/processed/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()  # Close to prevent display issues
print(" Plots saved to data/processed/eda_plots.png")

# 5. Save enhanced dataset
df.to_csv('data/processed/features_listings.csv', index=False)
print("\n Enhanced dataset saved to data/processed/features_listings.csv")

# 6. Generate insights report
with open('data/processed/insights_report.txt', 'w') as f:
    f.write("NAIROBI PROPERTY MARKET INSIGHTS\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"Total Listings: {len(df):,}\n")
    f.write(f"Average Price: KES {df['price'].mean():,.0f}\n")
    f.write(f"Median Price: KES {df['price'].median():,.0f}\n")
    f.write(f"Price Range: KES {df['price'].min():,.0f} - KES {df['price'].max():,.0f}\n\n")
    
    f.write("TOP 5 MOST EXPENSIVE LOCATIONS:\n")
    for loc, row in loc_stats.head(5).iterrows():
        f.write(f"  {loc}: KES {row['median']:,.0f} (avg: KES {row['mean']:,.0f}, n={int(row['count'])})\n")
    
    f.write("\nTOP 5 MOST AFFORDABLE LOCATIONS:\n")
    for loc, row in loc_stats.tail(5).iterrows():
        f.write(f"  {loc}: KES {row['median']:,.0f} (avg: KES {row['mean']:,.0f}, n={int(row['count'])})\n")
    
    f.write(f"\nAverage Price per sqm: KES {df['price_per_sqm'].mean():,.0f}\n")
    f.write(f"Average Price per Bedroom: KES {df['price_per_bedroom'].mean():,.0f}\n")

print("\n Insights report saved to data/processed/insights_report.txt")

# Print summary statistics
print("\n Summary Statistics:")
print(df[['price', 'bedrooms', 'bathrooms', 'size_sqm']].describe().round(0))