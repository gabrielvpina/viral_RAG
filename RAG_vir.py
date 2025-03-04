import json
import os
import sys
import torch
import logging
import ollama
from typing import List, Dict, Any, Optional
from colorama import init, Fore, Style
import warnings
import contextlib
import io

# Suppress specific warnings
import logging as pylogging
pylogging.getLogger('langchain').setLevel(pylogging.ERROR)
pylogging.getLogger('urllib3').setLevel(pylogging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class InteractiveVirologyRAGChat:
    def __init__(
        self,
        ollama_model="llama3.1:8b",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        db_path="virology_fulltext_faiss_index",
        max_context_history=5
    ):
        """
        Initialize an interactive RAG chatbot for scientific literature with streaming response
        
        Args:
            ollama_model (str): Ollama language model 
            embedding_model (str): Embedding model for semantic search
            db_path (str): Path to the vector database
            max_context_history (int): Maximum conversation context to retain
        """
        # Suppress initial warnings during initialization
        with contextlib.redirect_stderr(io.StringIO()):
            # Configure logging
            logging.basicConfig(
                level=logging.ERROR,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler("rag_chat.log"),
                    logging.StreamHandler()
                ]
            )
            
            # Device configuration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Model and embedding configurations
            self.ollama_model = ollama_model
            self.max_context_history = max_context_history
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": self.device}
            )
            
            # Text splitting configuration
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Conversation management
            self.conversation_history: List[Dict[str, Any]] = []
            
            # Vector database
            self.db_path = db_path
            self.vector_store = None
            
            # Load vector database
            self._load_vector_db()
    
    def _load_vector_db(self):
        """Load the vector database from disk"""
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                self.vector_store = FAISS.load_local(
                    self.db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            print(Fore.GREEN + "✓ Vector database loaded successfully")
        except Exception as e:
            print(Fore.RED + f"Error loading vector database: {str(e)}")
            self.vector_store = None
    
    def _retrieve_context(self, query: str, num_docs: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve contextually relevant scientific documents
        
        Args:
            query (str): User's query
            num_docs (int): Number of documents to retrieve
        
        Returns:
            List of contextually relevant documents
        """
        if not self.vector_store:
            print(Fore.RED + "Error: Vector database not loaded")
            return []
        
        # Retrieve documents with semantic search
        with contextlib.redirect_stderr(io.StringIO()):
            relevant_docs = self.vector_store.similarity_search(query, k=num_docs * 3)
        
        # Process and deduplicate documents
        processed_docs = []
        seen_pmids = set()
        
        for doc in relevant_docs:
            pmid = doc.metadata.get('pmid', '')
            if pmid not in seen_pmids and len(processed_docs) < num_docs:
                seen_pmids.add(pmid)
                processed_docs.append({
                    "pmid": pmid,
                    "title": doc.metadata.get('title', 'Unknown Title'),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        return processed_docs
    
    def _format_citations(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format citations for retrieved documents
        
        Args:
            documents (List[Dict]): Retrieved scientific documents
        
        Returns:
            Formatted citation string
        """
        if not documents:
            return "No citations found."
        
        citations = []
        for i, doc in enumerate(documents, 1):
            citation = (
                f"{i}. {doc['title']} "
                f"({doc['metadata'].get('authors', 'Unknown Authors')}, "
                f"{doc['metadata'].get('journal', 'Unknown Journal')}, "
                f"{doc['metadata'].get('publication_date', 'Unknown Date')})"
            )
            citations.append(citation)
        
        return "\n".join(citations)
    
    def _generate_enhanced_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive, context-aware prompt
        
        Args:
            query (str): User's query
            documents (List[Dict]): Retrieved scientific documents
        
        Returns:
            Formatted prompt for the LLM
        """
        # Prepare document context
        document_context = "\n\n".join([
            f"Document {i+1} Content:\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # Conversation history context
        history_context = "\n".join([
            f"Previous Query: {entry['query']}\nPrevious Response: {entry['response']}"
            for entry in self.conversation_history[-self.max_context_history:]
        ])
        
        # Enhanced prompt template
        prompt = f"""<|begin_of_text|><|system|>
        You are an advanced scientific research assistant specializing in academic literature.
        Provide precise, scholarly responses based on the retrieved scientific documents.
        Always cite specific sources and document numbers when referencing information.
        
        Conversation Context:
        {history_context if history_context else "No previous context"}

        Retrieved Scientific Documents:
        {document_context}

        <|user|>
        {query}
        <|assistant|>"""
        
        return prompt
    
    def _stream_ollama_response(self, prompt: str):
        """
        Stream the Ollama response in real-time
        
        Args:
            prompt (str): Formatted prompt for the LLM
        
        Yields:
            str: Streaming response tokens
        """
        full_response = ""
        for chunk in ollama.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "num_predict": 1024,
                "temperature": 0.7
            }
        ):
            if chunk.get('done', False):
                break
            
            if 'message' in chunk and 'content' in chunk['message']:
                token = chunk['message']['content']
                full_response += token
                yield token
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Process a conversational query with real-time retrieval and response
        
        Args:
            query (str): User's query
        
        Returns:
            Comprehensive query result with citations and context
        """
        try:
            # Retrieve contextually relevant documents
            retrieved_docs = self._retrieve_context(query)
            
            if not retrieved_docs:
                return {
                    "query": query,
                    "response": "No relevant scientific documents found.",
                    "citations": "",
                    "documents": []
                }
            
            # Generate enhanced prompt
            prompt = self._generate_enhanced_prompt(query, retrieved_docs)
            
            # Prepare response and citations
            response_parts = []
            
            # Store response for conversation history
            full_response = ""
            
            # Create citations
            citations = self._format_citations(retrieved_docs)
            
            # Stream response
            for token in self._stream_ollama_response(prompt):
                response_parts.append(token)
                full_response += token
                print(Fore.MAGENTA + token, end='', flush=True)
            
            # Print newline after streaming
            print()
            
            # Store conversation context
            conversation_entry = {
                "query": query,
                "response": full_response,
                "documents": retrieved_docs
            }
            self.conversation_history.append(conversation_entry)
            
            # Trim conversation history if needed
            if len(self.conversation_history) > self.max_context_history:
                self.conversation_history = self.conversation_history[-self.max_context_history:]
            
            return {
                "query": query,
                "response": full_response,
                "citations": citations,
                "documents": retrieved_docs
            }
        
        except Exception as e:
            print(Fore.RED + f"Error in chat processing: {str(e)}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "citations": "",
                "documents": []
            }
    
    def interactive_chat(self):
        """
        Start an interactive chat session with real-time input and response
        """
        print(Fore.GREEN + "🔬 Virology Research Assistant 🔬")
        print(Fore.YELLOW + "Type 'exit' to end the conversation.\n")
        
        while True:
            try:
                # User input
                user_query = input(Fore.CYAN + "You: ")
                
                # Exit condition
                if user_query.lower() in ['exit', 'quit', 'bye']:
                    print(Fore.GREEN + "Goodbye! Have a great day in research.")
                    break
                
                # Process query
                result = self.chat(user_query)
                
                # Display citations
                print(Fore.YELLOW + "\n📚 Relevant Citations:")
                print(Style.RESET_ALL + result['citations'])
                
                # Optional: Display document details
                print(Fore.GREEN + "\n🔍 Referenced Documents:")
                for i, doc in enumerate(result['documents'], 1):
                    print(Fore.BLUE + f"Document {i}:")
                    print(Style.RESET_ALL + f"  Title: {doc['title']}")
                    print(f"  Source: {doc['metadata'].get('journal', 'Unknown')}")
                    print(f"  Publication Date: {doc['metadata'].get('publication_date', 'Unknown')}")
                
                print("\n" + "-"*50 + "\n")
            
            except KeyboardInterrupt:
                print(Fore.GREEN + "\nChat interrupted. Type 'exit' to quit.")
            except Exception as e:
                print(Fore.RED + f"An error occurred: {str(e)}")

def main():
    # Suppress warnings during initialization
    with contextlib.redirect_stderr(io.StringIO()):
        # Initialize the interactive RAG chatbot
        rag_chat = InteractiveVirologyRAGChat(
            ollama_model="llama3.1:8b",
            db_path="virology_fulltext_faiss_index"
        )
    
    # Start interactive chat
    rag_chat.interactive_chat()

if __name__ == "__main__":
    main()
