from pathlib import Path 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#Project Path
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE = BASE_DIR / "knowledge_base"
DB_PATH = BASE_DIR/ "chroma_db"

# Split the documents into chunks 
splitter = RecursiveCharacterTextSplitter(
chunk_size=300,
chunk_overlap=80
)

documents = []

for file in KNOWLEDGE_BASE.glob("*.txt"):
    content = file.read_text(encoding="utf-8")
    chunks = splitter.split_text(content)

for file in KNOWLEDGE_BASE.glob("*.txt"):
    content = file.read_text(encoding="utf-8")
    chunks = splitter.split_text(content)

for index, chunk in enumerate(chunks):
    documents.append({
        "content": chunk,
        "source": file.name,
        "chunk_id": index
    })
print("Total chunks:", len(documents))


#Step 2: Load the embedding model

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

#Step 3: Create the Chroma vector database

vector_store = Chroma(
    collection_name = "resume_knowledge",
    embedding_function = embeddings,
    persist_directory = str(DB_PATH)
)

#Step 4: Store the chunks and their metadata
vector_store.add_texts(
texts=[doc["content"] for doc in documents],
metadatas=[
{
"source": doc["source"],
"chunk_id": doc["chunk_id"]
}
for doc in documents
],
ids=[
f'{doc["source"]}_{doc["chunk_id"]}'
for doc in documents
]
)

print("Vector database created successfully!")
print("Database location:", DB_PATH)