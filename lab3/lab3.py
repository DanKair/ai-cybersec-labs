import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("self_imporvement_dataset.csv")


""" Task 1 """
# Display the first 5 rows
print(df.head())

# Number of rows and columns
print(df.shape)

# Information about columns and data types
print(df.info())

# Data types of all columns
print(df.dtypes)

""" Task 2 """
# Counts all mising values
print(df.isnull().sum())

# Shows sum of all duplicated values
print(df.duplicated().sum())

"""Task 3 """

print(df["label"].unique())

print(df["label"].value_counts())

print(df["label"].value_counts(normalize=True) * 100)

# Task 4
print(df.describe())

"""Task 5"""

# Class Distribution
df["label"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()

# Histogram example
df["views_count"].hist()

plt.xlabel("Views")
plt.ylabel("Frequency")
plt.title("Views Count Distribution")
plt.show()

# Scatter plot
plt.scatter(
    df["day_since_published"],
    df["views_count"]
)

plt.xlabel("Days Since Published")
plt.ylabel("Views Count")
plt.title("Days Since Published vs Views")
plt.show()

# Comparative analysis of classes
print(
    df.groupby("label")[[
    "title_length",
    "day_since_published",
    "views_count"
]].mean()
)

# Correlation analysis
print(
    df[[
        "title_length",
        "day_since_published",
        "views_count"
    ]].corr()
)

df.to_csv("clean_dataset.csv", index=False)

