from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np 
import faiss
import pandas as pd
from tqdm import tqdm
import csv
from sklearn.decomposition import PCA
import time
import os
import kagglehub

# If dataset not downloaded, grab it from kaggle
print("Grabbing DiverseVul-Cleaned.csv dataset")
path = kagglehub.dataset_download("ahmedtabib/diversevul-cleaned")
path = path + "/DiverseVul-Cleaned.csv"
print("Dataset path: ", path)

# import dataset into dataframe, keep specific columns
df_divul = pd.read_csv(path, low_memory=False)
df_divul = df_divul.drop(columns=['Unnamed: 0', 'commit_id', 'project', 'hash', 'size', 'target'])

# Seperate bad code into a dataframe
df_bad = df_divul[df_divul['cwe'] != '[]']
df_bad_msg = df_bad.drop(columns=['func', 'cwe'])
df_bad = df_bad.reset_index(drop=True)

# Seperate good code into a dataframe
df_good = df_divul[df_divul['cwe'] == '[]']
df_good_msg = df_good.drop(columns=['func', 'cwe'])
df_good = df_good.reset_index(drop=True)

# Try to seperate into good/bad code csv files
df_good_msg.to_csv("good.csv", index=False)
df_bad_msg.to_csv("bad.csv", index=False)

# ----- Embedding Sections -----
# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# FAISS index (using L2 similarity for this example)
index = faiss.IndexFlatL2(768) # 768 for the dimention

# Process rows and generate embeddings
for col, row in tqdm(df_bad.iterrows(), total=len(df_bad), desc="Processing rows"):
    # Tokenize code
    # Tokenize and process row
    inputs = tokenizer.encode_plus(
        row['func'], 
        max_length=512, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )

    tokens_tensor = inputs["input_ids"]

    # Generate embeddings
    with torch.no_grad():  # Disable gradients for efficiency
        context_embeddings = model(tokens_tensor)[0]  # Get hidden states
        cls_embedding = context_embeddings[:, 0, :]  # Extract the [CLS] embedding
    
    print(f"CLS embedding shape: {cls_embedding.shape}")
    # Convert to numpy and add to FAISS index
    cls_embedding_np = cls_embedding.squeeze(0).cpu().numpy()  # Shape: (embedding_dim,)
    index.add(cls_embedding_np[np.newaxis, :])  # Add as a row to the FAISS index

    # Save FAISS index to a file
    faiss.write_index(index, "faiss_index.bin")
    
# Save the FAISS index to a file
faiss.write_index(index, "faiss_index.bin")
