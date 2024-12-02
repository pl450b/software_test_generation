from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm

df = pd.read_json("hf://datasets/CyberNative/Code_Vulnerability_Security_DPO/secure_programming_dpo.json", lines=True)

df_cpp = df[df['lang'] == 'c++']
df_cpp_reduced = df_cpp.drop(columns=['system', 'question'])
df_cpp_reduced = df_cpp_reduced.reset_index(drop=True)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

embeddings_list = []

for index, row in tqdm(df_cpp_reduced.iterrows(), total=len(df_cpp_reduced), desc="Processing rows"):
    # Tokenize natural language and code
    nl_tokens = tokenizer.tokenize(row['vulnerability'])
    code_tokens = tokenizer.tokenize(row['chosen'])

    if(df_cpp_reduced[','] == 127):
        break

    # Combine tokens with special tokens
    tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]

    # Convert tokens to token IDs
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Generate embeddings using the model
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]

    # Print the vector embedding
    embeddings_list.append(context_embeddings)

    print(context_embeddings)

# Add all the vectors we made to the dataframe
df_cpp_reduced['vector'] = embeddings_list
df_cpp_reduced.to_csv("df_cpp_embeddings.csv", index=True)
print(df_cpp_reduced[['vulnerability', 'vector']].head())