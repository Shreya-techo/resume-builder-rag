from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
knowledge_base = Path(__file__).parent / "knowledge_base"

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 80
)

all_chunks = []
for file in knowledge_base.glob("*.txt"):
    content = file.read_text(encoding = "utf-8")
    chunks = splitter.split_text(content)

    for index, chunk in enumerate(chunks):
        all_chunks.append({
        "source": file.name,
        "chunk_id": index,
        "content": chunk
    })
print("Total chunks:", len(all_chunks))

for chunk in all_chunks:
    print("\n" + "=" * 50)
    print("Source:", chunk["source"])
    print("Chunk ID:", chunk["chunk_id"])
    print(chunk["content"])
