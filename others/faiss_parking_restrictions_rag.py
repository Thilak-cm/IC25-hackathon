import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the CSV
df = pd.read_csv("data/Parking Restrictions.csv")
df.drop(df.index[103:], inplace=True)

# Select multiple columns for embedding
columns_to_encode = ["Parking Lot / Zone Name", "Restrictions", "Required", "Parking Restrictions", "Overflow Lot"]

# Fill NaN values with an empty string and concatenate selected columns
df["combined_text"] = df[columns_to_encode].fillna("").agg(" | ".join, axis=1)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for each row
embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)

# Store metadata (row info)
metadata = df.to_dict(orient="records")

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index & metadata
faiss.write_index(index, "parking_restrictions.index")
np.save("metadata.npy", metadata)

print("FAISS index & metadata saved!")