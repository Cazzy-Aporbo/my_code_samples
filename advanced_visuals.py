import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Generate a fun and unique dataset
def generate_data(n=500):
    np.random.seed(42)
    random.seed(42)
    
    categories = ["Technology", "Art", "Music", "Sports", "Science", "Fashion"]
    subcategories = {
        "Technology": ["AI", "Blockchain", "Cybersecurity", "Quantum Computing"],
        "Art": ["Painting", "Sculpture", "Digital Art", "Photography"],
        "Music": ["Jazz", "Rock", "Classical", "Electronic"],
        "Sports": ["Basketball", "Soccer", "Tennis", "F1"],
        "Science": ["Physics", "Biology", "Astronomy", "Genetics"],
        "Fashion": ["Streetwear", "Haute Couture", "Sustainable", "Athleisure"]
    }
    
    data = []
    for _ in range(n):
        category = random.choice(categories)
        subcategory = random.choice(subcategories[category])
        value1 = np.random.normal(loc=50, scale=20)
        value2 = np.random.uniform(10, 100)
        value3 = np.random.randint(1, 100)
        data.append([category, subcategory, value1, value2, value3])
    
    return pd.DataFrame(data, columns=["Category", "Subcategory", "Popularity", "Engagement", "Uniqueness"])

# Load data
df = generate_data(1000)

# Create a unique and colorful visualization
plt.figure(figsize=(14, 8))
sns.set_palette("husl")
sns.set_style("whitegrid")

# Violin Plot without `split=True`
sns.violinplot(
    data=df, x="Category", y="Popularity", hue="Subcategory", 
    inner="quartile", linewidth=1.5, palette="coolwarm"
)
plt.title("Popularity Distribution Across Categories and Subcategories", fontsize=14, fontweight="bold")
plt.xlabel("Category", fontsize=12)
plt.ylabel("Popularity Score", fontsize=12)
plt.xticks(rotation=15)
plt.legend(title="Subcategory", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Pairplot for interactive insights
sns.pairplot(df, hue="Category", palette="viridis")
plt.show()

# Hexbin Heatmap for Engagement vs. Uniqueness
plt.figure(figsize=(10, 6))
hb = plt.hexbin(df["Engagement"], df["Uniqueness"], gridsize=30, cmap="inferno", mincnt=1)
plt.colorbar(hb, label="Density of Observations")
plt.xlabel("Engagement Level")
plt.ylabel("Uniqueness Score")
plt.title("Engagement vs. Uniqueness Heatmap", fontsize=14, fontweight="bold")
plt.show()

# KDEplot with 2D Density Estimate
plt.figure(figsize=(12, 7))
sns.kdeplot(data=df, x="Engagement", y="Popularity", fill=True, cmap="coolwarm", levels=30)
plt.title("2D Density Estimate of Engagement vs. Popularity", fontsize=14, fontweight="bold")
plt.xlabel("Engagement Level")
plt.ylabel("Popularity Score")
plt.show()
