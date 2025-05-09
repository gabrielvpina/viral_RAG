import json
import pandas as pd
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def analyze_viral_sequences(json_file_path, taxonomy_data_path, output_json_path=None):
    """
    Analyze viral sequences from a JSON file using Ollama LLM to generate AI summaries.
    
    Args:
        json_file_path (str): Path to the JSON file containing viral sequence data
        taxonomy_data_path (str): Path to the CSV file with viral taxonomy information
        output_json_path (str, optional): Path to save the updated JSON. If None, will overwrite input file.
    
    Returns:
        dict: The updated JSON data with AI summaries
    """
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Load the viral taxonomy data
    taxonomy_df = pd.read_csv(taxonomy_data_path)
    
    # Initialize the LLM
    model = OllamaLLM(model="llama3.2:latest")
    
    # Create a template for the prompt
    template = """
    You are an expert in virology and will analyze some information of potential viral sequences 
    from a bioinformatics analysis. Here is the information:
    
    {sequence_data}
    
    Here is the information about the viral order/family/genus:
    {taxonomy_info}
    
    Based on this information, make a text (maximum of 200 words) about the taxonomy information (hosts), 
    and evaluate if the sequence above is a known virus, a novel virus, or a non-viral sequence. 
    Talk about the demarcation criteria of the taxonomy group.
    """
    
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
                    "HMM_Hits": hmm_hits,
                    "ORF_Data": orf_data
                }
                
                # Convert to string without the sequence data
                complete_data_str = json.dumps(complete_data, indent=2)
                
                # Generate AI summary
                result = chain.invoke({
                    "sequence_data": complete_data_str,
                    "taxonomy_info": taxonomy_info
                })
                
                viral_hit["AI_summary"] = result.strip()
                print(f"Generated AI summary for {query_id}")
            else:
                print(f"No matching taxonomy information found for {query_id}")
    
    # Save the updated JSON
    if output_json_path is None:
        output_json_path = json_file_path
        
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Updated JSON saved to {output_json_path}")
    return data

# Example usage
if __name__ == "__main__":
    # Define paths
    json_file = "data.json"  # Path to your JSON file
    taxonomy_file = "viral_taxonomy_data.csv"  # Path to your taxonomy CSV file
    output_file = "viral_analysis_results.json"  # Path for output
    
    # Run the analysis
    updated_data = analyze_viral_sequences(json_file, taxonomy_file, output_file)
