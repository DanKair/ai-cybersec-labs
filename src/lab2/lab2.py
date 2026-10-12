import pandas as pd

# Task 9: Load the dataset
# Make sure 'self_improvement_dataset.csv' is uploaded to your environment
df = pd.read_csv("self_imporvement_dataset.csv")

# Display the first few rows to confirm the pipeline worked
print("--- FIRST 5 ROWS ---")
print(df.head())

# Task 10: Generic info (Data types, non-null counts, memory usage)
print("\n--- DATASET STRUCTURE & INFO ---")
df.info()

# Summary statistics for numeric features (title_length, days_published, views_count)
print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe())

# Task 11: Target variable balance check
print("\n--- TARGET LABEL DISTRIBUTION ---")
print(df['label'].value_counts())

print("\n--- TARGET LABEL PERCENTAGES ---")
print(df['label'].value_counts(normalize=True) * 100)