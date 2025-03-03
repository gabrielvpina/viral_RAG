import json
import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("vector_creation.log"),
                              logging.StreamHandler()])

def create_vector_database(
    input_file_path,
    output_db_path="virology_fulltext_faiss_index",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=100
):
    """Create a vector database from articles JSON file and save it"""
    
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Initialize embedding model
    logging.info(f"Initializing embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )
    
    # Test embedding model
    logging.info("Testing embedding model...")
    test_embedding = embeddings.embed_documents(["This is a test document"])
    logging.info(f"Test embedding dimension: {len(test_embedding[0])}")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Load articles
    logging.info(f"Loading articles from {input_file_path}")
    try:
        with open(input_file_path, 'r') as f:
            articles = json.load(f)
        logging.info(f"Loaded {len(articles)} articles")
    except Exception as e:
        logging.error(f"Failed to load articles: {str(e)}")
        return False
    
    if not articles:
        logging.error("No articles found in the input file")
        return False
    
    # Prepare documents for indexing
    documents = []
    
    for article in tqdm(articles, desc="Processing articles"):
        try:
            # Skip articles without basic information
            if not all(key in article for key in ["pmid", "title", "abstract"]):
                logging.warning(f"Skipping article missing required fields")
                continue
            
            # Prepare basic metadata
            metadata = {
                "pmid": article["pmid"],
                "title": article["title"],
                "category": article.get("category", "Unknown"),
                "authors": ", ".join(article["authors"]) if isinstance(article.get("authors", []), list) else article.get("authors", "Unknown"),
                "journal": article.get("journal", "Unknown"),
                "publication_date": article.get("publication_date", "Unknown"),
                "keywords": ", ".join(article["keywords"]) if isinstance(article.get("keywords", []), list) else article.get("keywords", ""),
                "doi": article.get("doi", ""),
                "pmc_id": article.get("pmc_id", "")
            }
            
            # Process full text if available
            if article.get("has_fulltext", False) and article.get("full_text"):
                full_text = article["full_text"]
                
                # Create metadata for full text
                full_text_metadata = metadata.copy()
                full_text_metadata["source_type"] = "full_text"
                
                # Create document chunks from full text
                chunks = text_splitter.create_documents([full_text], [full_text_metadata])
                documents.extend(chunks)
                
                # Add title and abstract as a separate chunk
                title_abstract = f"Title: {article['title']}\nAbstract: {article['abstract']}"
                title_abstract_metadata = metadata.copy()
                title_abstract_metadata["source_type"] = "title_abstract"
                title_abstract_chunks = text_splitter.create_documents([title_abstract], [title_abstract_metadata])
                documents.extend(title_abstract_chunks)
            
            else:
                # If no full text, use title and abstract
                content = f"Title: {article['title']}\nAbstract: {article['abstract']}"
                metadata["source_type"] = "abstract_only"
                chunks = text_splitter.create_documents([content], [metadata])
                documents.extend(chunks)
        
        except Exception as e:
            logging.error(f"Error processing article {article.get('pmid', 'unknown')}: {str(e)}")
    
    logging.info(f"Created {len(documents)} document chunks from {len(articles)} articles")
    
    if not documents:
        logging.error("No valid documents created. Cannot build vector store.")
        return False
    
    # Create and save vector store
    try:
        logging.info("Creating FAISS index...")
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save to disk
        vector_store.save_local(output_db_path)
        logging.info(f"Vector database successfully saved to {output_db_path}")
        return True
    
    except Exception as e:
        logging.error(f"Error creating vector database: {str(e)}")
        # Check first document for debugging
        if documents:
            first_doc = documents[0]
            logging.error(f"First document sample: {first_doc.page_content[:100]}...")
            logging.error(f"First document metadata: {first_doc.metadata}")
        return False

if __name__ == "__main__":
    # Set your paths
    INPUT_FILE = "virology_papers/virology_papers_fulltext.json"
    OUTPUT_DB = "virology_fulltext_faiss_index"
    
    # Create the vector database
    success = create_vector_database(
        input_file_path=INPUT_FILE,
        output_db_path=OUTPUT_DB
    )
    
    if success:
        print("Vector database created successfully!")
    else:
        print("Failed to create vector database. Check logs for details.")
