from langchain_ollama import ChatOllama

# Initialize the model connecting to local Ollama
llm = ChatOllama(
    model="llama3.1",
    temperature=0,  # Important: Set to 0 for reproducible research results
    base_url="http://localhost:11434"
)

print("Querying local Llama 3.1...")
response = llm.invoke("Explain quantum mechanics in one sentence.")
print(f"Response: {response.content}")