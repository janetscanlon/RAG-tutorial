import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """load all the files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    #check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files")
    
    #load all txt files from the directory 
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0: 
        raise FileNotFoundError(f"No files found in {docs_path}. Please add your company documents")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    Content Length: {len(doc.page_content)} characters:")
        print(f"    Preview: {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0): 
    """split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)

    if chunks: 

        # for i, chunk in enumerate(chunks[:5]):
        #     print(f"\n--- chunk {i+1} ---")
        #     print(f"Source: {chunk.metadata['source']}")
        #     print(f"Length: {len(chunk.page_content)} characters:")
        #     print(f"Content:")
        #     print(chunk.page_content)
        #     print("-"* 50)

        if len(chunks) > 5: 
            print(f"\n... and {len(chunks)-5} more chunks")

        return chunks
    
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #Create ChromaDB vector store 
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore
    
def main():
    #load source documents
    documents = load_documents(docs_path="docs")
    #chunk documents 
    chunks = split_documents(documents)
    #embed and store 
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()