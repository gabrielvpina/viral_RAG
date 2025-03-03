import json
import os
import torch
import numpy as np
import logging
import ollama
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("rag_system.log"),
                              logging.StreamHandler()])

class VirologyRAG:
    def __init__(
        self,
        ollama_model="llama3.2",
        device="cuda",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        db_path="virology_fulltext_faiss_index"
    ):
        """Initialize the RAG system with a local LLM and embedding model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        # Store ollama model name
        self.ollama_model = ollama_model
        
        # Initialize embeddings model for vector search
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device}
        )
        
        # Initialize text splitter for chunking (used in query processing)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store
        self.db_path = db_path
        self.vector_store = None
        
        # Try to load the vector database
        if os.path.exists(db_path):
            self.load_vector_db()
        else:
            logging.error(f"Vector database not found at {db_path}")
            logging.info("Please create the vector database first using the create_vector_database.py script")
    
    def load_vector_db(self):
        """Load the vector database from disk"""
        try:
            logging.info(f"Loading vector database from {self.db_path}")
            self.vector_store = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            logging.info("Vector database loaded successfully")
        except Exception as e:
            logging.error(f"Error loading vector database: {str(e)}")
            self.vector_store = None
    
    def create_vector_db(self, classified_articles_path):
        """Create a vector database from classified articles with full text"""
        # Load classified articles
        with open(classified_articles_path, 'r') as f:
            articles = json.load(f)
        
        if not articles:
            logging.error("No articles found in the input file")
            return
            
        # Prepare documents for indexing
        documents = []
        
        for article in tqdm(articles, desc="Processing articles for vector DB"):
            try:
                # Verify article has required fields
                required_fields = ["pmid", "title", "abstract"]
                if not all(field in article for field in required_fields):
                    logging.warning(f"Article missing required fields: {article.get('pmid', 'unknown')}")
                    continue
                    
                # Determine what content to index
                if article.get("has_fulltext", False) and article.get("full_text"):
                    # Process full text in chunks
                    full_text = article["full_text"]
                    
                    # Create metadata
                    metadata = {
                        "pmid": article["pmid"],
                        "title": article["title"],
                        "category": article.get("category", "Unknown"),
                        "authors": ", ".join(article["authors"]) if isinstance(article["authors"], list) else article.get("authors", "Unknown"),
                        "journal": article.get("journal", "Unknown"),
                        "publication_date": article.get("publication_date", "Unknown"),
                        #"keywords": ", ".join(article["keywords"]) if isinstance(article.get("keywords", []), list) else article.get("keywords", ""),
                        "doi": article.get("doi", ""),
                        "pmc_id": article.get("pmc_id", ""),
                        "source_type": "full_text"
                    }
                    
                    # Create document chunks from full text
                    chunks = self.text_splitter.create_documents([full_text], [metadata])
                    
                    # Add title and abstract as a separate chunk for better retrieval
                    title_abstract = f"Title: {article['title']}\nAbstract: {article['abstract']}"
                    title_abstract_metadata = metadata.copy()
                    title_abstract_metadata["source_type"] = "title_abstract"
                    title_abstract_chunk = self.text_splitter.create_documents([title_abstract], [title_abstract_metadata])
                    
                    # Add all chunks
                    documents.extend(chunks)
                    documents.extend(title_abstract_chunk)
                    
                else:
                    # If no full text, just use title and abstract
                    content = f"Title: {article['title']}\nAbstract: {article['abstract']}"
                    
                    metadata = {
                        "pmid": article["pmid"],
                        "title": article["title"],
                        "category": article.get("category", "Unknown"),
                        "authors": ", ".join(article["authors"]) if isinstance(article.get("authors", []), list) else article.get("authors", "Unknown"),
                        "journal": article.get("journal", "Unknown"),
                        "publication_date": article.get("publication_date", "Unknown"),
                        #"keywords": ", ".join(article["keywords"]) if isinstance(article.get("keywords", []), list) else article.get("keywords", ""),
                        "doi": article.get("doi", ""),
                        "pmc_id": article.get("pmc_id", ""),
                        "source_type": "abstract_only"
                    }
                    
                    chunks = self.text_splitter.create_documents([content], [metadata])
                    documents.extend(chunks)
            
            except Exception as e:
                logging.error(f"Error processing article {article.get('pmid', 'unknown')} for vector DB: {str(e)}")
        
        logging.info(f"Created {len(documents)} document chunks from {len(articles)} articles")
        
        if len(documents) == 0:
            logging.error("No valid documents created from articles. Cannot create vector store.")
            return
        
        try:
            # Create vector store
            logging.info("Creating FAISS index...")
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            
            # Save vector store
            self.vector_store.save_local(self.db_path)
            logging.info(f"Vector database saved to {self.db_path}")
        except Exception as e:
            logging.error(f"Error creating vector database: {str(e)}")
            # For debugging
            if documents:
                first_doc = documents[0]
                logging.error(f"First document sample: {first_doc.page_content[:100]}...")
                logging.error(f"Metadata sample: {first_doc.metadata}")
    
    def generate_prompt(self, query, relevant_docs, num_docs=3):
        """Generate a prompt for the LLM using the query and retrieved documents"""
        # Process and format the retrieved chunks
        formatted_docs = []
        seen_pmids = set()
        
        for i, doc in enumerate(relevant_docs):
            pmid = doc.metadata['pmid']
            
            # Skip duplicate articles but include different chunks from the same article
            if pmid in seen_pmids and doc.metadata.get('source_type') != 'title_abstract':
                continue
            
            # For the first occurrence of an article, include its metadata
            if pmid not in seen_pmids:
                doc_header = (
                    f"Document {len(formatted_docs) + 1}:\n"
                    f"Title: {doc.metadata['title']}\n"
                    f"Category: {doc.metadata['category']}\n"
                    f"Journal: {doc.metadata['journal']} ({doc.metadata['publication_date']})\n"
                )
                seen_pmids.add(pmid)
            else:
                doc_header = f"Additional content from Document {list(seen_pmids).index(pmid) + 1}:\n"
            
            # Format the content based on source type
            if doc.metadata.get('source_type') == 'title_abstract':
                doc_content = doc.page_content
            elif doc.metadata.get('source_type') == 'abstract_only':
                doc_content = doc.page_content
            else:
                # For full text chunks, add a more descriptive header
                doc_content = f"Text excerpt: {doc.page_content}"
            
            formatted_docs.append(f"{doc_header}{doc_content}")
            
            # Limit to num_docs unique articles
            if len(seen_pmids) >= num_docs:
                break
        
        context = "\n\n".join(formatted_docs)
        
        prompt = f"""<|begin_of_text|><|system|>
                You are a helpful virology research assistant with access to a database of scientific articles.
                You will be given a query about virology and relevant scientific articles from the database.
                Provide a comprehensive, accurate answer based on the information in the articles.
                Always cite the article titles when using information from them.
                If the articles don't contain relevant information to answer the query, state that clearly.
                <|user|>
                Query: {query}

                Relevant articles:
                {context}
                <|assistant|>"""
        
        return prompt
    
    def query(self, query, num_docs=3, max_tokens=1024, temperature=0.7, top_p=0.9):
        """Query the RAG system with a user question"""
        try:
            if not self.vector_store:
                logging.error("Vector database not loaded or created")
                return {
                    "query": query,
                    "response": "Error: Vector database not loaded or created. Please initialize the system with articles first.",
                    "used_articles": [],
                    "num_total_docs_retrieved": 0,
                    "num_unique_articles_used": 0,
                    "error": True
                }

            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=num_docs*3)
            
            if not relevant_docs:
                logging.warning("No relevant documents found for query")
                return {
                    "query": query,
                    "response": "No relevant documents found for your query. Please try a different question.",
                    "used_articles": [],
                    "num_total_docs_retrieved": 0,
                    "num_unique_articles_used": 0
                }
            
            # Generate prompt with context
            prompt = self.generate_prompt(query, relevant_docs, num_docs)

            # Get response from Ollama
            logging.info(f"Sending query to Ollama model: {self.ollama_model}")
            response_data = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            )

            # Extract response
            if "message" in response_data and "content" in response_data["message"]:
                response = response_data["message"]["content"]
            else:
                logging.error(f"Error in model response: {response_data}")
                return {
                    "query": query,
                    "response": "Error generating response.",
                    "used_articles": [],
                    "num_total_docs_retrieved": len(relevant_docs),
                    "num_unique_articles_used": 0,
                    "error": True
                }

            # Extract just the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[1].strip()
            
            # Collect unique articles that were used
            used_articles = []
            seen_pmids = set()
            
            for doc in relevant_docs:
                pmid = doc.metadata['pmid']
                if pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    used_articles.append({
                        "pmid": pmid,
                        "title": doc.metadata['title'],
                        "authors": doc.metadata['authors'],
                        "journal": doc.metadata['journal'],
                        "publication_date": doc.metadata['publication_date'],
                        "category": doc.metadata.get('category', 'Unknown'),
                        "doi": doc.metadata.get('doi', ""),
                        "source_type": doc.metadata.get('source_type', "unknown")
                    })
                    
                    # Limit to num_docs unique articles
                    if len(used_articles) >= num_docs:
                        break
            
            # Return the response and used articles as a dictionary
            return {
                "query": query,
                "response": response,
                "used_articles": used_articles,
                "num_total_docs_retrieved": len(relevant_docs),
                "num_unique_articles_used": len(used_articles)
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            # Return a dictionary instead of a string to maintain consistent return type
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "used_articles": [],
                "num_total_docs_retrieved": 0,
                "num_unique_articles_used": 0,
                "error": True
            }

if __name__ == "__main__":
    # Initialize the RAG system by loading the pre-built vector database
    rag = VirologyRAG(
        ollama_model="llama3.2",
        db_path="virology_fulltext_faiss_index"  # Path to your pre-built database
    )
    
    if rag.vector_store is None:
        print("Error: Vector database could not be loaded.")
        print("Please run the create_vector_database.py script first.")
        exit(1)
    
    # Example query
    result = rag.query("What are the AI (artificial intelligence) tools available in Virology?")
    
    # Check if result has error
    if result.get("error", False):
        print(f"Error: {result['response']}")
    else:
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Used articles: {len(result['used_articles'])}")
        for article in result['used_articles']:
            print(f"- {article['title']} ({article['journal']}, {article['publication_date']})")