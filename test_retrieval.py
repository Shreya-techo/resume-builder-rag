from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"

# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to the existing vector database
vector_store = Chroma(
    collection_name="resume_knowledge",
    embedding_function=embeddings,
    persist_directory=str(DB_PATH)
)

# Ask a question
query = "What are the key points a resume should have?"

# Retrieve the three most relevant chunks
results = vector_store.similarity_search(query, k=3)

print("\nQuestion:", query)

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print("Source:", document.metadata["source"])
    print("Content:", document.page_content)