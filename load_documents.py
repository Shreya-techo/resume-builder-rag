from pathlib import Path
knowledge_base = Path(__file__).parent/ "knowledge_base"

# find all text files
files = list(knowledge_base.glob("*.txt"))

print("Number of files found: ", len(files))

# Read each file
documents = []

for file in files:
    content = file.read_text(encoding="utf-8")

    documents.append({
        "filename": file.name,
        "content": content
    })
    print(f"\nLoaded: {file.name}")
    print("Characters:" , len(content))

print("\nAll documents loaded successfully!")
