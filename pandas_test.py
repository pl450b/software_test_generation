import pandas as pd

df = pd.read_json("hf://datasets/CyberNative/Code_Vulnerability_Security_DPO/secure_programming_dpo.json", lines=True)

# Only keep columns we're interested in
df_cpp = df[df['lang'] == 'c++']
df_cpp_reduced = df_cpp.drop(columns=['system', 'question'])
df_cpp_reduced = df_cpp_reduced.reset_index(drop=True)

#Output to JSON and CSV
df_cpp_reduced.to_json("df_cpp_reduced.json", orient="records", lines=True)
df_cpp_reduced.to_csv("df_cpp_reduced.csv", index=True)

# Classification loop
for index, row in df_cpp_reduced.iterrows():
    print(f"Vulnerability: {row['vulnerability']}, Code: {row['lang']}")

df_cpp_reduced.to_csv("df_cpp_indexed.csv", index=True)

print(df_cpp_reduced)

