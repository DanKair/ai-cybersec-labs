import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('data/ssh_anomaly_dataset.csv')

# 2. Basic dimensions (Task 2)
print("--- Task 2: Initial Analysis ---")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Display data types, missing counts, and unique values
info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Unique Values': df.nunique(),
    'Missing Values': df.isnull().sum()
})
print("\nDataset Structure:")
print(info_df)

# Show basic statistical summaries
print("\nBasic Statistical Characteristics:")
print(df.describe(include='all'))


print("--- Task 3: Data Quality Analysis ---")
# Identify duplicates
duplicates_count = df.duplicated().sum()
print(f"Number of duplicate records: {duplicates_count}")

# Drop duplicates if any exist
if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# Check for constant features (variance = 0 or unique values = 1)
constant_features = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Constant features to drop: {constant_features}")
df = df.drop(columns=constant_features)

# Handle missing values (Example: Fill numerical with median)
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print("--- Task 4: Cyber Threat Analysis ---")
target_col = 'event_type'

class_counts = df[target_col].value_counts()
class_percentages = df[target_col].value_counts(normalize=True) * 100

print("Class Distribution:")
for cls in class_counts.index:
    print(f"Class '{cls}': {class_counts[cls]} objects ({class_percentages[cls]:.2f}%)")

# Calculate Imbalance Ratio (Majority Class / Minority Class)
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

print("--- Tasks 5 & 6: Statistical & Outlier Analysis ---")

# Choose 3 critical numerical features from your dataset to test for outliers
df['timestamp'] = pd.to_datetime(df['timestamp'])

for col in ['status', 'event_type', 'username', 'source_ip']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# --- TASK 8: FEATURE ENGINEERING (Moved up to build metrics first) ---
print("--- Task 8: Feature Engineering ---")
# Feature 1: IP connection counts
df['ip_appearance_count'] = df.groupby('source_ip')['source_ip'].transform('count')

# Feature 2: Failed login tally per IP
df['is_failed'] = df['status'].apply(lambda x: 1 if any(f in x for f in ['fail', 'invalid', 'error']) else 0)
df['ip_failed_attempts'] = df.groupby('source_ip')['is_failed'].transform('sum')

# Feature 3: Target user diversity
df['ip_user_diversity'] = df.groupby('source_ip')['username'].transform('nunique')

# Feature 4: Time of day activity
df['hour'] = df['timestamp'].dt.hour
df['hourly_log_volume'] = df.groupby('hour')['hour'].transform('count')

# --- TASK 4: CYBER THREAT ANALYSIS (Creating the Target Label) ---
print("\n--- Task 4: Cyber Threat Analysis (Label Engineering) ---")
# Heuristic rule: If an IP has > 10 failed logins OR targets more than 2 distinct users, label as Attack (1)
df['is_attack'] = np.where((df['ip_failed_attempts'] > 10) | (df['ip_user_diversity'] > 2), 1, 0)

target_col = 'is_attack'
class_counts = df[target_col].value_counts()
class_percentages = df[target_col].value_counts(normalize=True) * 100

print("Class Distribution:")
for cls in class_counts.index:
    print(f"Class '{cls}' (0=Benign, 1=Attack): {class_counts[cls]} entries ({class_percentages[cls]:.2f}%)")

imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

# --- TASK 5 & 6: STATISTICAL & OUTLIER ANALYSIS ---
print("\n--- Tasks 5 & 6: Statistical & Outlier Analysis ---")
features_to_check = ['ip_appearance_count', 'ip_failed_attempts', 'ip_user_diversity']

for col in features_to_check:
    # Calculations for Task 5
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\nFeature: {col}")
    print(f"Mean: {df[col].mean():.2f} | Median: {df[col].median():.2f} | Std Dev: {df[col].std():.2f}")
    print(f"Q1: {Q1} | Q3: {Q3} | IQR: {IQR}")
    print(f"Detected Outliers: {len(outliers)}")

# --- TASK 7: CORRELATION ANALYSIS ---
print("\n--- Task 7: Correlation Analysis ---")
numeric_cols = ['ip_appearance_count', 'ip_failed_attempts', 'ip_user_diversity', 'hourly_log_volume', 'is_attack']
corr_matrix = df[numeric_cols].corr()
print("Correlation Matrix:")
print(corr_matrix)


# --- TASK 9 & 10: FEATURE SELECTION & VISUALIZATION ---
print("\n--- Task 10: Generating Visualizations ---")

# 1. Class Distribution (FIXED)
plt.figure(figsize=(5,4))
sns.countplot(x='is_attack', data=df, hue='is_attack', palette='Set2', legend=False)
plt.title('SSH Class Distribution (0: Benign, 1: Attack)')
plt.xlabel('Class Label')
plt.ylabel('Log Entry Count')
plt.savefig('images/viz_1_class_dist.png', bbox_inches='tight')
plt.close()

# 2. Histogram
plt.figure(figsize=(5,4))
sns.histplot(df['ip_failed_attempts'], bins=20, kde=True, color='crimson')
plt.title('Histogram of Failed Attempts per IP')
plt.savefig('images/viz_2_histogram.png', bbox_inches='tight')
plt.close()

# 3. Boxplot (FIXED)
plt.figure(figsize=(5,4))
sns.boxplot(x='is_attack', y='ip_user_diversity', data=df, hue='is_attack', palette='Pastel1', legend=False)
plt.title('Boxplot of Targeted User Diversity')
plt.savefig('images/viz_3_boxplot.png', bbox_inches='tight')
plt.close()

# 4. Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('images/viz_4_heatmap.png', bbox_inches='tight')
plt.close()

# 5. Scatter Plot
plt.figure(figsize=(5,4))
sns.scatterplot(x='ip_appearance_count', y='ip_failed_attempts', hue='is_attack', data=df)
plt.title('Scatter Plot: Connection Attempts vs Failures')
plt.savefig('images/viz_5_scatter.png', bbox_inches='tight')
plt.close()

print("Visualizations successfully saved as PNG files with clean console logs!")


# --- TASK 11: DATASET PREPARATION SAVE ---
df.to_csv('data/cleaned/prepared_dataset.csv', index=False)
print("\n--- Task 11: Dataset successfully saved to 'prepared_dataset.csv' ---")