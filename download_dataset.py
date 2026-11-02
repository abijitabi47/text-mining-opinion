
from datasets import load_dataset
from pathlib import Path
import pandas as pd

print("Loading product review dataset...")

dataset = load_dataset("debojit01/fake-review-dataset")

print("\nAvailable splits:", list(dataset.keys()))

# Use the training split
df = dataset["train"].to_pandas()

print("\nAvailable columns:")
print(df.columns.tolist())

# The dataset contains text_ and rating columns
text_col = "text_" if "text_" in df.columns else "text"

df = df[[text_col, "rating"]].dropna()
df = df.rename(columns={text_col: "review"})

# Select up to 10,000 reviews
df = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

# Derive sentiment from star ratings
def classify_sentiment(rating):
    rating = float(rating)

    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(classify_sentiment)

# Save the dataset
output = Path("data/reviews.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False, encoding="utf-8")

print("\nDataset saved successfully!")
print("Total reviews:", len(df))
print("File size (MB):", round(output.stat().st_size / (1024 ** 2), 2))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())
