import pandas as pd
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama3-8B-Instruct-Replete-Adapted-GGUF",
	filename="Llama3-8B-Instruct-Replete-Adapted-IQ2_M.gguf",
)

df = pd.read_json("hf://datasets/CyberNative/Code_Vulnerability_Security_DPO/secure_programming_dpo.json", lines=True)

df_cpp = df[df['lang'] == 'c++']
df_cpp_reduced = df_cpp.drop(columns=['system', 'question'])

print(df_cpp_reduced)

# Chat loop
conversation_history = []
print("\nStarting chat with the LLM. Type 'exit' to end the chat.")

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chat ended.")
        break
    
    # Append the user's message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Generate a response
    response = llm.create_chat_completion(messages=conversation_history)
    
    # Extract and print the model's response
    llm_response = response['choices'][0]['message']['content']
    print(f"LLM: {llm_response}")
    
    # Add the LLM's response to the conversation history
    conversation_history.append({"role": "assistant", "content": llm_response})
