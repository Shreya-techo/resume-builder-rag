from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# 1. Project paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"

# 2. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Connect to the existing ChromaDB
vector_store = Chroma(
    collection_name="resume_knowledge",
    embedding_function=embeddings,
    persist_directory=str(DB_PATH)
)

# 4. Create the retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 5. Connect to the local Ollama model
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# 6. Ask a question
query = input("\nAsk a resume-related question: ")

# 7. Retrieve relevant knowledge
documents = retriever.invoke(query)

context = "\n\n".join(
    document.page_content for document in documents
)

# 8. Give the retrieved knowledge to the LLM
prompt = f"""
You are a careful resume-writing assistant.

Answer the user's question using the supplied knowledge-base context.

STRICT RULES:

1. Use the context as your source of resume-writing guidance.

2. Never invent facts, achievements, statistics, qualifications, or experience.

3. Do not present fictional examples as real achievements.

4. If you provide an example, label it clearly as hypothetical.

5. If a question requires personal information that has not been provided,
explain what information is missing.

6. Do not generate a complete fictional project example unless the user
explicitly requests an illustrative example.

7. Never include percentages, numerical improvements, business outcomes,
or project achievements unless they are explicitly supported by
information provided by the user.

8. When the user asks for general advice, provide instructions or a
checklist instead of inventing a project scenario.

9. Do not claim that the knowledge base contains information that it does not.

10. Give a clear, practical answer.

KNOWLEDGE-BASE CONTEXT:
{context}

USER QUESTION:
{query}
"""

response = llm.invoke(prompt)

print("\n--- Retrieved Knowledge ---")

for document in documents:
    print(f"\nSource: {document.metadata['source']}")
    print(document.page_content)

print("\n--- RAG Answer ---")
print(response.content)