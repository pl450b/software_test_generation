import pandas as pd
import ollama

# Pull dataset I found
df = pd.read_json("hf://datasets/CyberNative/Code_Vulnerability_Security_DPO/secure_programming_dpo.json", lines=True)

# Only select code/vulnerabilites in C++
df_cpp = df[df['lang'] == 'c++']
df_cpp_reduced = df_cpp.drop(columns=['system', 'question'])
df_cpp_reduced = df_cpp_reduced.reset_index(drop=True)

# Classification loop
for vuln in df_cpp_reduced['vulnerability']:
    
    msg = "Give me a single phrase classifing the vulnerability in the folowing: " + vuln

    response = ollama.chat(model='llama3.2:3b', messages=[
    {
        'role': 'user',
        'content': msg,
    },
    ])
    print("[VULNERABILIY] ", vuln)
    print("[RESPONSE]", response['message']['content'])
    

