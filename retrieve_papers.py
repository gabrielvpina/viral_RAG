import requests
import json
import time
import os
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import logging
import concurrent.futures
import io
import re
import PyPDF2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fulltext_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullTextExtractor:
    def __init__(self, output_dir="virology_papers", email="your.email@example.com"):
        """
        Initialize the full text extractor
        
        Args:
            output_dir (str): Directory to save files
            email (str): Your email for API courtesy
        """
        self.output_dir = output_dir
        self.email = email
        self.session = requests.Session()
        
        # Set up User-Agent header for API requests
        self.headers = {
            'User-Agent': f'FullTextExtractor/1.0 ({email})'
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create subdirectories for PDFs
        self.pdf_dir = os.path.join(output_dir, "pdfs")
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)
    
    def search_europe_pmc(self, query="virology", page_size=100, max_results=200):
        """
        Search for papers using the Europe PMC API with pagination
        
        Args:
            query (str): Search term
            page_size (int): Number of results per page
            max_results (int): Maximum number of results to retrieve
            
        Returns:
            list: Papers with metadata
        """
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        
        # Add open access filter and virology-related terms
        full_query = f"({query}) AND OPEN_ACCESS:Y"
        
        all_papers = []
        cursor = "*"
        
        logger.info(f"Searching Europe PMC for '{full_query}'...")
        
        while len(all_papers) < max_results:
            params = {
                "query": full_query,
                "resultType": "core",
                "cursorMark": cursor,
                "pageSize": page_size,
                "format": "json"
            }
            
            try:
                response = self.session.get(base_url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                next_cursor = data.get("nextCursorMark", "")
                result_list = data.get("resultList", {}).get("result", [])
                
                logger.info(f"Found {len(result_list)} papers in this batch")
                
                # Process this batch of results
                for item in result_list:
                    paper_info = {
                        "pmid": item.get("pmid", ""),
                        "pmcid": item.get("pmcid", ""),
                        "doi": item.get("doi", ""),
                        "title": item.get("title", "Untitled"),
                        "journal": item.get("journalTitle", "Unknown Journal"),
                        "publication_date": item.get("firstPublicationDate", ""),
                        "abstract": item.get("abstractText", ""),
                        "authors": [author.get("fullName", "") for author in item.get("authorList", {}).get("author", [])],
                        "keywords": item.get("keywordList", {}).get("keyword", [])
                    }
                    all_papers.append(paper_info)
                
                # Check if we have enough papers or reached the end
                if len(all_papers) >= max_results or cursor == next_cursor or not next_cursor:
                    break
                
                cursor = next_cursor
                
                # Be nice to the API
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error in Europe PMC API request: {e}")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON response from Europe PMC API")
                break
        
        # Trim to max_results if we got more
        if len(all_papers) > max_results:
            all_papers = all_papers[:max_results]
            
        logger.info(f"Retrieved metadata for {len(all_papers)} papers")
        return all_papers
    
    def get_full_text_urls(self, papers):
        """
        Get full text URLs for papers from Europe PMC
        
        Args:
            papers (list): List of paper dictionaries
            
        Returns:
            list: Papers with full text URLs
        """
        papers_with_urls = []
        
        logger.info(f"Getting full text URLs for {len(papers)} papers...")
        
        for paper in tqdm(papers, desc="Finding full text URLs"):
            # Initialize URLs as empty
            paper["fulltext_xml_url"] = None
            paper["fulltext_pdf_url"] = None
            paper["has_fulltext"] = False
            
            # Try to get URLs based on PMCID (best source for full text)
            if paper.get("pmcid"):
                pmcid = paper["pmcid"].replace("PMC", "")
                
                # XML URL
                paper["fulltext_xml_url"] = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                
                # PDF URL
                paper["fulltext_pdf_url"] = f"https://europepmc.org/articles/PMC{pmcid}/pdf/document.pdf"
                
                paper["has_fulltext"] = True
            
            # If no PMCID but has DOI, try to get from Unpaywall
            elif paper.get("doi") and not paper.get("has_fulltext"):
                try:
                    doi = paper["doi"]
                    url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
                    response = self.session.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Get best open access URL
                        best_oa_url = None
                        if data.get("best_oa_location"):
                            best_oa_url = data["best_oa_location"].get("url_for_pdf") or data["best_oa_location"].get("url")
                        
                        # If no best_oa_location, check other OA locations
                        if not best_oa_url and data.get("oa_locations"):
                            for location in data["oa_locations"]:
                                pdf_url = location.get("url_for_pdf")
                                if pdf_url:
                                    best_oa_url = pdf_url
                                    break
                                elif location.get("url"):
                                    best_oa_url = location.get("url")
                                    break
                        
                        if best_oa_url:
                            if best_oa_url.lower().endswith('.pdf'):
                                paper["fulltext_pdf_url"] = best_oa_url
                            else:
                                paper["fulltext_html_url"] = best_oa_url
                            
                            paper["has_fulltext"] = True
                    
                    # Be nice to the API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error checking Unpaywall for DOI {doi}: {e}")
            
            papers_with_urls.append(paper)
        
        # Count papers with full text URLs
        papers_with_fulltext = sum(1 for p in papers_with_urls if p.get("has_fulltext"))
        logger.info(f"Found {papers_with_fulltext} papers with full text URLs out of {len(papers_with_urls)}")
        
        return papers_with_urls
    
    def download_and_extract_fulltext(self, papers, max_workers=5):
        """
        Download and extract full text content for papers
        
        Args:
            papers (list): List of paper dictionaries with URLs
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            list: Papers with extracted full text
        """
        papers_with_fulltext = []
        
        # Count papers with full text URLs
        papers_to_process = [p for p in papers if p.get("has_fulltext")]
        logger.info(f"Downloading and extracting full text for {len(papers_to_process)} papers...")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {executor.submit(self.process_paper, paper): paper for paper in papers_to_process}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_paper), total=len(papers_to_process), desc="Processing papers"):
                paper = future_to_paper[future]
                try:
                    processed_paper = future.result()
                    papers_with_fulltext.append(processed_paper)
                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('pmid', '')} - {paper.get('doi', '')}: {e}")
                    papers_with_fulltext.append(paper)  # Add the paper without full text
        
        # Add papers that didn't have full text URLs
        papers_without_fulltext = [p for p in papers if not p.get("has_fulltext")]
        papers_with_fulltext.extend(papers_without_fulltext)
        
        # Count papers with extracted full text
        successful_extractions = sum(1 for p in papers_with_fulltext if p.get("fulltext") is not None)
        logger.info(f"Successfully extracted full text for {successful_extractions} out of {len(papers_with_fulltext)} papers")
        
        return papers_with_fulltext
    
    def process_paper(self, paper):
        """
        Process a single paper to extract its full text
        
        Args:
            paper (dict): Paper information dictionary
            
        Returns:
            dict: Paper with extracted full text
        """
        # Try XML first if available (best structured content)
        if paper.get("fulltext_xml_url"):
            try:
                response = self.session.get(paper["fulltext_xml_url"], headers=self.headers)
                if response.status_code == 200:
                    xml_content = response.text
                    fulltext = self.extract_text_from_xml(xml_content)
                    if fulltext:
                        paper["fulltext"] = fulltext
                        paper["fulltext_source"] = "xml"
                        return paper
            except Exception as e:
                logger.error(f"Error extracting text from XML for {paper.get('pmid', '')}: {e}")
        
        # Try PDF if XML failed or wasn't available
        if paper.get("fulltext_pdf_url"):
            try:
                response = self.session.get(paper["fulltext_pdf_url"], headers=self.headers, stream=True)
                if response.status_code == 200:
                    # Save PDF
                    filename = f"{paper.get('pmid', '')}_" + (paper.get('doi', '').replace('/', '_') if paper.get('doi') else 'unknown')
                    pdf_path = os.path.join(self.pdf_dir, f"{filename}.pdf")
                    
                    with open(pdf_path, 'wb') as pdf_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            pdf_file.write(chunk)
                    
                    # Extract text from PDF
                    fulltext = self.extract_text_from_pdf(pdf_path)
                    if fulltext:
                        paper["fulltext"] = fulltext
                        paper["fulltext_source"] = "pdf"
                        paper["pdf_path"] = pdf_path
                        return paper
            except Exception as e:
                logger.error(f"Error extracting text from PDF for {paper.get('pmid', '')}: {e}")
        
        # Try HTML if other methods failed
        if paper.get("fulltext_html_url"):
            try:
                response = self.session.get(paper["fulltext_html_url"], headers=self.headers)
                if response.status_code == 200:
                    html_content = response.text
                    fulltext = self.extract_text_from_html(html_content)
                    if fulltext:
                        paper["fulltext"] = fulltext
                        paper["fulltext_source"] = "html"
                        return paper
            except Exception as e:
                logger.error(f"Error extracting text from HTML for {paper.get('pmid', '')}: {e}")
        
        # If all methods failed
        paper["fulltext"] = None
        paper["fulltext_source"] = None
        return paper
    
    def extract_text_from_xml(self, xml_text):
        """
        Extract text content from PMC XML
        
        Args:
            xml_text (str): XML content
            
        Returns:
            str: Extracted text content
        """
        try:
            soup = BeautifulSoup(xml_text, "xml")
            
            # Extract article title
            title_elem = soup.find("article-title")
            title = title_elem.get_text() if title_elem else ""
            
            full_text = []
            if title:
                full_text.append(f"# {title}\n")
            
            # Extract abstract
            abstract_elems = soup.find_all("abstract")
            for abstract in abstract_elems:
                abstract_text = abstract.get_text().strip()
                if abstract_text:
                    full_text.append("## Abstract\n")
                    full_text.append(abstract_text + "\n")
            
            # Extract sections and paragraphs
            sections = soup.find_all(["sec", "body"])
            
            for section in sections:
                # Get section title
                section_title = section.find("title")
                if section_title:
                    heading = section_title.get_text().strip()
                    if heading:
                        full_text.append(f"## {heading}\n")
                
                # Get paragraphs
                paragraphs = section.find_all("p")
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text:
                        full_text.append(text + "\n")
            
            # If we didn't find structured content, extract all text
            if len(full_text) <= 2:  # Just title and maybe abstract
                body = soup.find("body")
                if body:
                    full_text.append(body.get_text().strip())
            
            return "\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text())
                
                return "\n\n".join(text)
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return None
    
    def extract_text_from_html(self, html_content):
        """
        Extract text from HTML content
        
        Args:
            html_content (str): HTML content
            
        Returns:
            str: Extracted text
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Try to find the main content
            main_content = None
            
            # Try different selectors that might contain the article text
            for selector in ['article', 'main', '.content', '.article', '#content', '#main']:
                content = soup.select(selector)
                if content:
                    main_content = content[0]
                    break
            
            # If no main content found, use the whole body
            if not main_content:
                main_content = soup.body
            
            # Extract text
            if main_content:
                # Get all headings and paragraphs
                elements = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
                
                text = []
                for element in elements:
                    if element.name.startswith('h'):
                        # Add header formatting
                        level = int(element.name[1])
                        prefix = '#' * level
                        text.append(f"{prefix} {element.get_text().strip()}")
                    else:
                        # Regular paragraph
                        text.append(element.get_text().strip())
                
                return "\n\n".join(text)
            
            # If structured extraction failed, get all text
            return soup.get_text()
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return None
    
    def save_to_json(self, papers, filename="virology_papers_fulltext.json"):
        """
        Save papers with full text to JSON
        
        Args:
            papers (list): List of paper dictionaries
            filename (str): Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        logger.info(f"Saving {len(papers)} papers to {output_path}")
        
        # Prepare papers for JSON export
        export_papers = []
        for paper in papers:
            # Create a clean copy without file paths
            paper_export = paper.copy()
            
            # Remove local file paths for security
            if "pdf_path" in paper_export:
                paper_export.pop("pdf_path")
                
            export_papers.append(paper_export)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_papers, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved papers to {output_path}")
        
        # Also save a summary CSV
        self.save_summary_csv(papers)
    
    def save_summary_csv(self, papers):
        """
        Save a summary of papers as CSV for quick reference
        
        Args:
            papers (list): List of paper dictionaries
        """
        summary_data = []
        
        for paper in papers:
            fulltext_length = len(paper.get("fulltext", "")) if paper.get("fulltext") else 0
            
            summary_data.append({
                "pmid": paper.get("pmid", ""),
                "pmcid": paper.get("pmcid", ""),
                "doi": paper.get("doi", ""),
                "title": paper.get("title", ""),
                "journal": paper.get("journal", ""),
                "publication_date": paper.get("publication_date", ""),
                "has_fulltext": paper.get("fulltext") is not None,
                "fulltext_source": paper.get("fulltext_source", "none"),
                "abstract_length": len(paper.get("abstract", "")),
                "fulltext_length": fulltext_length
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.output_dir, "paper_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to {csv_path}")
    
    def run(self, query="virology", max_results=30):
        """
        Run the full paper retrieval and parsing pipeline
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of papers to retrieve
        """
        # Step 1: Search for papers
        papers = self.search_europe_pmc(query, max_results=max_results)
        
        if not papers:
            logger.error("No papers found. Exiting.")
            return []
        
        # Step 2: Get full text URLs
        papers_with_urls = self.get_full_text_urls(papers)
        
        # Step 3: Download and extract full text
        papers_with_fulltext = self.download_and_extract_fulltext(papers_with_urls)
        
        # Step 4: Save to JSON
        self.save_to_json(papers_with_fulltext)
        
        return papers_with_fulltext

# Example usage
if __name__ == "__main__":
    extractor = FullTextExtractor(
        output_dir="virology_papers",
        email="gvprodrigues.ppggbm@uesc.br"  # Replace with your email
    )
    
    # Run with a specific query
    papers = extractor.run(
        query="virology AND viral AND viruses",
        max_results=50  # Adjust based on your needs
    )
    
    # Print summary
    if papers:
        with_fulltext = sum(1 for p in papers if p.get("fulltext") is not None)
        print(f"\nProcessed {len(papers)} papers")
        print(f"Papers with full text: {with_fulltext} ({with_fulltext/len(papers)*100:.1f}%)")
        
        # Show sources of full text
        sources = {}
        for paper in papers:
            source = paper.get("fulltext_source")
            if source:
                sources[source] = sources.get(source, 0) + 1
        
        print("\nFull text sources:")
        for source, count in sources.items():
            print(f"  - {source}: {count} papers")