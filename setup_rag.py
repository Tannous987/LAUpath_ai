"""
RAG Setup Script for LAUpath AI

This script processes PDF documents from the data/lau_documents directory and creates
a vector database using Chroma for retrieval-augmented generation.

Run this script once before using the main application to set up the vector database.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PDF_DIRECTORY = "./data/lau_documents"
VECTOR_DB_DIRECTORY = "./vector_db"
COLLECTION_NAME = "lau_documents"
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def setup_rag() -> None:
    """
    Set up the RAG system by processing PDFs and creating a vector database.
    
    This function:
    1. Loads all PDF files from the data/lau_documents directory
    2. Splits documents into chunks
    3. Creates embeddings and stores them in Chroma vector database
    """
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    # Initialize embeddings
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    
    # Get all PDF files
    pdf_dir = Path(PDF_DIRECTORY)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIRECTORY}")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIRECTORY}")
    
    print(f"Found {len(pdf_files)} PDF file(s) to process...")
    
    # Load and process all PDFs
    all_documents = []
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}...")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add metadata to identify source
            for doc in documents:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["source_path"] = str(pdf_file)
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {str(e)}")
            continue
    
    if not all_documents:
        raise ValueError("No documents were successfully loaded from PDF files.")
    
    print(f"\nTotal pages loaded: {len(all_documents)}")
    
    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} document chunks")
    
    # Create or update vector database
    print(f"\nCreating vector database at {VECTOR_DB_DIRECTORY}...")
    
    # Remove existing database if it exists
    if os.path.exists(VECTOR_DB_DIRECTORY):
        import shutil
        print("Removing existing vector database...")
        shutil.rmtree(VECTOR_DB_DIRECTORY)
    
    # Create new vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"✓ Vector database created successfully!")
    print(f"  - Location: {VECTOR_DB_DIRECTORY}")
    print(f"  - Collection: {COLLECTION_NAME}")
    print(f"  - Total chunks: {len(chunks)}")
    print("\nRAG setup complete! You can now run the main application.")


if __name__ == "__main__":
    try:
        setup_rag()
    except Exception as e:
        print(f"\n✗ Error during RAG setup: {str(e)}")
        raise

