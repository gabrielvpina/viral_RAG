import json
import pandas as pd
import os
import re
import argparse
from langchain_core.prompts import ChatPromptTemplate

def remove_thinking_tags(text):
    """
    Remove <thinking>...</thinking> tags and their content from the text.
    
    Args:
        text: Text that may contain thinking tags (string or LangChain message object)
        
    Returns:
        str: Cleaned text without thinking sections
    """
    # Handle different response types (AIMessage, ChatMessage, string, etc.)
    if hasattr(text, 'content'):
        # For LangChain message objects (AIMessage, ChatMessage, etc.)
        text_content = text.content
    elif isinstance(text, dict) and 'content' in text:
        # For dictionary-like objects with content key
        text_content = text['content']
    else:
        # Try to convert to string if it's another type
        text_content = str(text)
    
    # Remove <thinking>...</thinking> blocks
    cleaned_text = re.sub(r'<thinking>.*?</thinking>', '', text_content, flags=re.DOTALL)
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
    
    # Also handle variations like <Thinking>...</Thinking> (case insensitive)
    cleaned_text = re.sub(r'<[Tt]hinking>.*?</[Tt]hinking>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<[Tt]hink>.*?</[Tt]hink>', '', cleaned_text, flags=re.DOTALL)
    
    # Remove any extra whitespace that might be left
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def get_llm_model(args):
    """
    Initialize and return the appropriate LLM based on user arguments.
    
    Args:
        args: Command line arguments parsed by argparse
        
    Returns:
        An initialized LLM model
    """
    # For Ollama models (local)
    if args.model_type == "ollama":
        from langchain_ollama.llms import OllamaLLM
        # Ollama returns string directly, so wrap it in a simple class for consistent handling
        ollama = OllamaLLM(model=args.model_name)
        
        # Wrap the Ollama model to ensure consistent behavior with chat models
        from langchain_core.language_models.llms import LLM
        from typing import Any, List, Optional
        
        class OllamaWrapper(LLM):
            llm: Any
            
            def __init__(self, llm):
                super().__init__()
                self.llm = llm
                
            def _call(self, prompt: str, **kwargs) -> str:
                return self.llm.invoke(prompt)
            
            @property
            def _llm_type(self) -> str:
                return "ollama_wrapper"
                
        return OllamaWrapper(ollama)
    
    # For OpenAI models (API-based)
    elif args.model_type == "openai":
        from langchain_openai import ChatOpenAI
        if not args.api_key:
            raise ValueError("API key is required for OpenAI models")
        
        os.environ["OPENAI_API_KEY"] = args.api_key
        return ChatOpenAI(model_name=args.model_name, temperature=0.3)
    
    # For Anthropic models (API-based)
    elif args.model_type == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not args.api_key:
            raise ValueError("API key is required for Anthropic models")
        
        os.environ["ANTHROPIC_API_KEY"] = args.api_key
        return ChatAnthropic(model_name=args.model_name, temperature=0.3)
    
    # For Google models (API-based)
    elif args.model_type == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not args.api_key:
            raise ValueError("API key is required for Google models")
        
        os.environ["GOOGLE_API_KEY"] = args.api_key
        return ChatGoogleGenerativeAI(model=args.model_name, temperature=0.3)
        
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

def analyze_viral_sequences(args):
    """
    Analyze viral sequences from a JSON file using the specified LLM to generate AI summaries.
    
    Args:
        args: Command line arguments parsed by argparse
    
    Returns:
        dict: The updated JSON data with AI summaries
    """
    json_file_path = args.input_json
    taxonomy_data_path = args.taxonomy_data
    output_json_path = args.output_json if args.output_json else json_file_path
    
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Load the viral taxonomy data
    taxonomy_df = pd.read_csv(taxonomy_data_path)
    
    # Initialize the LLM
    try:
        model = get_llm_model(args)
    except ValueError as e:
        print(f"Error initializing model: {e}")
        return None
    
    # Create a template for the prompt
    template = """
    You are an expert in virology and will analyze some information of potential viral sequences from a bioinformatics analysis.
    You need to be skeptical in your analysis. 
    Here is the information:
    
    {sequence_data}
    
    Here is the information about the viral order/family/genus:
    {taxonomy_info}
    
    Based on this information, make a text (maximum of 200 words) about the taxonomy information (hosts), 
    and evaluate if the sequence above is a known virus, a novel virus, or a non-viral sequence. 
    Talk about the demarcation criteria of the taxonomy group. 
    Remember, the analyses also depends of the results in BLASTx and BLASTn identity and Coverage, 
    normally a known virus has more than 90% in aminoacid/nucleotide similarity and at least 70% of coverage.
    """
    
    # Handle different model types appropriately
    if args.model_type == "ollama":
        # For Ollama models, we use the wrapper directly
        chain = model
    else:
        # For chat models (OpenAI, Anthropic, Google), we use ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
    
    # Process each viral hit
    if "Viral_Hits" in data:
        for i, viral_hit in enumerate(data["Viral_Hits"]):
            query_id = viral_hit.get("QueryID")
            
            # Create AI_summary field if it doesn't exist
            if "AI_summary" not in viral_hit:
                viral_hit["AI_summary"] = ""
            
            # Extract taxonomic information
            taxonomy_info = None
            
            # Try to find the most specific taxonomy information in this order: Genus, Family, Order
            for tax_level in ["Genus", "Family", "Order"]:
                if viral_hit.get(tax_level) and not pd.isna(viral_hit.get(tax_level)) and viral_hit.get(tax_level) != "":
                    matching_rows = taxonomy_df[(taxonomy_df['type'] == tax_level) & 
                                              (taxonomy_df['name'] == viral_hit.get(tax_level))]
                    if not matching_rows.empty:
                        taxonomy_info = matching_rows.iloc[0]['info']
                        break
            
            # If we found taxonomy info, generate AI summary
            if taxonomy_info:
                # Prepare sequence data (exclude FullSeq to avoid token limit issues)
                sequence_data = {k: v for k, v in viral_hit.items() if k != "FullSeq"}
                
                # Find related HMM hits for this QueryID
                hmm_hits = []
                if "HMM_hits" in data:
                    hmm_hits = [hit for hit in data["HMM_hits"] if hit.get("QueryID") == query_id]
                
                # Find related ORF data for this QueryID
                orf_data = []
                if "ORF_Data" in data and query_id in data["ORF_Data"]:
                    orf_data = data["ORF_Data"][query_id]
                
                # Prepare the complete data for this sequence
                complete_data = {
                    "Viral_Hit": sequence_data,
                    "HMM_Hits": hmm_hits
                    #"ORF_Data": orf_data
                }
                
                # Convert to string without the sequence data
                complete_data_str = json.dumps(complete_data, indent=2)
                
                # Generate AI summary
                if args.model_type == "ollama":
                    # For Ollama models, format the prompt as a string
                    formatted_prompt = template.format(
                        sequence_data=complete_data_str,
                        taxonomy_info=taxonomy_info
                    )
                    result = chain.invoke(formatted_prompt)
                else:
                    # For chat models, use the dictionary format
                    result = chain.invoke({
                        "sequence_data": complete_data_str,
                        "taxonomy_info": taxonomy_info
                    })
                
                # Remove thinking tags if present
                cleaned_result = remove_thinking_tags(result)
                
                viral_hit["AI_summary"] = cleaned_result.strip()
                print(f"Generated AI summary for {query_id}")
            else:
                print(f"No matching taxonomy information found for {query_id}")
    
    # Save the updated JSON
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Updated JSON saved to {output_json_path}")
    return data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze viral sequences using AI models')
    
    # Required arguments
    parser.add_argument('--input-json', required=True, help='Path to the input JSON file with viral sequence data')
    parser.add_argument('--taxonomy-data', required=True, help='Path to the CSV file with viral taxonomy information')
    
    # Model selection arguments
    parser.add_argument('--model-type', required=True, choices=['ollama', 'openai', 'anthropic', 'google'], 
                        help='Type of model to use for analysis')
    parser.add_argument('--model-name', required=True, 
                        help='Name of the model (e.g., "qwen3:4b" for ollama, "gpt-3.5-turbo" for OpenAI)')
    parser.add_argument('--api-key', help='API key for cloud models (required for OpenAI, Anthropic, Google)')
    
    # Optional arguments
    parser.add_argument('--output-json', help='Path to save the output JSON (defaults to overwriting input)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run analysis
    analyze_viral_sequences(args)

if __name__ == "__main__":
    main()