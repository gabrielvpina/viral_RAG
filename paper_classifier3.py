import json
import os
import logging
import time
import ollama
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("classification.log"),
                              logging.StreamHandler()])

class ViralArticleClassifier:
    def __init__(self, model_name="llama3.2", output_dir="classified_articles"):
        """Initialize the classifier with an Ollama model"""
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Categories for classification
        self.categories = [
            "Viruses in plants",
            "COVID-19 studies",
            "Fungal viruses",
            "Viral zoonoses",
            "Viruses in humans (excluding COVID-19)",
            "Insect viruses"
        ]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_prompt(self, title, abstract, fulltext=None):
        """Create a classification prompt for the LLM with optional full text"""
        #text_for_classification = fulltext if fulltext and fulltext != abstract else abstract
        text_for_classification = abstract
        
        max_chars = 3000  # Approximately 1000-1500 tokens
        if len(text_for_classification) > max_chars:
            text_for_classification = text_for_classification[:max_chars] + "..."
        
        prompt = f"""
You are an expert virologist that classifies scientific articles about viruses into categories.
Examine the article title, abstract, and available text, then classify it into exactly one of these categories:
1. Viruses in plants  
2. COVID-19 studies  
3. Fungal viruses  
4. Viral zoonoses  
5. Viruses in humans (excluding COVID-19)  
6. Insect viruses

Output only the category name, nothing else.

Title: {title}

Abstract: {abstract}

{'Full Text Excerpt: ' + text_for_classification if fulltext and fulltext != abstract else ''}
""".strip()
        return prompt
    
    def classify_article(self, title, abstract, fulltext=None):
        """Classify a single article using Ollama"""
        prompt = self.generate_prompt(title, abstract, fulltext)
        
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        category_response = response['message']['content'].strip()
        
        # Ensure the response matches one of our categories
        for category in self.categories:
            if category.lower() in category_response.lower():
                return category
        
        for category in self.categories:
            for word in category.lower().split():
                if word in category_response.lower() and len(word) > 3:
                    return category
        
        return f"{category}"
    
    def batch_classify_articles(self, articles_file, output_file=None, batch_size=5):
        """Classify a batch of articles from a JSON file and save them into category-specific files"""
        with open(articles_file, 'r') as f:
            articles = json.load(f)
        
        categorized_articles = {category: [] for category in self.categories}
        categorized_articles["Other Virology Subjects"] = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1}")
            
            for article in tqdm(batch, desc=f"Classifying articles {i+1}-{i+len(batch)}"):
                title = article["title"]
                abstract = article["abstract"]
                fulltext = article.get("fulltext", "")
                
                try:
                    if fulltext and len(fulltext) > len(abstract) * 1.5:
                        category = self.classify_article(title, abstract, fulltext)
                        used_fulltext = True
                    else:
                        category = self.classify_article(title, abstract)
                        used_fulltext = False
                    
                    result = {
                        "pmid": article["pmid"],
                        "title": title,
                        "abstract": abstract,
                        "has_fulltext": article.get("has_fulltext", False),
                        "used_fulltext_for_classification": used_fulltext,
                        "category": category,
                        "authors": article["authors"],
                        "journal": article["journal"],
                        "keywords": article["keywords"],
                        "publication_date": article["publication_date"],
                        "doi": article.get("doi", ""),
                        "pmc_id": article.get("pmc_id", ""),
                        "fulltext": fulltext
                    }
                    
                    categorized_articles[category].append(result)
                    logging.info(f"Classified '{title}' as '{category}'")
                
                except Exception as e:
                    logging.error(f"Error classifying article {article.get('pmid', 'unknown')}: {str(e)}")
            
            if i + batch_size < len(articles):
                logging.info("Pausing between batches...")
                time.sleep(5)
        
        # Save categorized articles into separate JSON files
        for category, articles_list in categorized_articles.items():
            category_filename = os.path.join(self.output_dir, f"{category.replace(' ', '_')}.json")
            with open(category_filename, 'w') as f:
                json.dump(articles_list, f, indent=2)
            logging.info(f"Saved {len(articles_list)} articles to {category_filename}")
        
        logging.info("Classification complete.")
        return categorized_articles

if __name__ == "__main__":
    classifier = ViralArticleClassifier()
    articles_file = "virology_papers/virology_papers_fulltext.json"
    classifier.batch_classify_articles(articles_file)
