from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load the embeddings (same embedding model as ingestion) and vector store
embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "Which island does SpaceX lease for its launches in the Pacific?"

# search db as the retriever and look for the top 3 chunks
retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#   search_type="similarity_score_threshold",
#   search_kwargs={
#           "k": 5,
#           "score_threshold": 0.3 # Only return chunks with cosine similarity at least 0.3
#       }
#   )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")