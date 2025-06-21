import json
import re
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPDFScraper:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.extracted_data = {
            "metadata": {},
            "acts": [],
            "statistics": {},
            "raw_pages": []
        }
    
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the PDF"""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                
                return {
                    "title": str(metadata.title) if metadata and metadata.title else "Unknown",
                    "author": str(metadata.author) if metadata and metadata.author else "Unknown",
                    "subject": str(metadata.subject) if metadata and metadata.subject else "Unknown",
                    "creator": str(metadata.creator) if metadata and metadata.creator else "Unknown",
                    "producer": str(metadata.producer) if metadata and metadata.producer else "Unknown",
                    "creation_date": str(metadata.creation_date) if metadata and metadata.creation_date else "Unknown",
                    "modification_date": str(metadata.modification_date) if metadata and metadata.modification_date else "Unknown",
                    "total_pages": len(reader.pages)
                }
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"error": str(e)}
    
    def parse_table_data(self, tables: List) -> List[Dict[str, Any]]:
        """Parse acts from table data"""
        acts = []
        
        for table in tables:
            if not table:
                continue
                
            # Skip header rows
            for row in table[2:]:  # Skip first 2 rows (headers)
                if len(row) >= 4:
                    section = row[0] if row[0] else ""
                    serial_no = row[1] if row[1] else ""
                    act_name = row[2] if row[2] else ""
                    year = row[3] if row[3] else ""
                    act_no = row[4] if len(row) > 4 and row[4] else ""
                    
                    # Clean up the data
                    if act_name and act_name.strip():
                        act_data = {
                            "section": section.strip(),
                            "serial_number": serial_no.strip().rstrip('.'),
                            "name": act_name.strip(),
                            "year": year.strip() if year and year.strip().isdigit() else None,
                            "act_number": act_no.strip() if act_no else None
                        }
                        
                        # Only add if we have a meaningful act name
                        if len(act_data["name"]) > 5:
                            acts.append(act_data)
        
        return acts
    
    def parse_text_data(self, text: str) -> List[Dict[str, Any]]:
        """Parse acts from text using improved regex patterns"""
        acts = []
        
        # Pattern to match the tabular format: Number. Act Name Year ActNo
        pattern = r'(\d+)\.\s+([^0-9]+?)\s+(\d{4})\s+(\d+)'
        
        matches = re.finditer(pattern, text, re.MULTILINE)
        
        for match in matches:
            serial_no = match.group(1)
            act_name = match.group(2).strip()
            year = match.group(3)
            act_no = match.group(4)
            
            # Clean up act name - remove extra whitespace and newlines
            act_name = re.sub(r'\s+', ' ', act_name)
            
            if len(act_name) > 5:  # Only include meaningful act names
                act_data = {
                    "serial_number": serial_no,
                    "name": act_name,
                    "year": year,
                    "act_number": act_no
                }
                acts.append(act_data)
        
        return acts
    
    def extract_content_and_acts(self) -> tuple:
        """Extract both raw content and parsed acts"""
        pages_content = []
        all_acts = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text()
                    
                    # Extract tables
                    tables = page.extract_tables()
                    
                    page_data = {
                        "page_number": page_num,
                        "text": text.strip() if text else "",
                        "tables": tables if tables else [],
                        "char_count": len(text) if text else 0,
                        "word_count": len(text.split()) if text else 0
                    }
                    
                    pages_content.append(page_data)
                    
                    # Parse acts from tables (preferred method)
                    if tables:
                        table_acts = self.parse_table_data(tables)
                        for act in table_acts:
                            act["page_number"] = page_num
                            act["source"] = "table"
                        all_acts.extend(table_acts)
                    
                    # Parse acts from text as fallback
                    if text:
                        text_acts = self.parse_text_data(text)
                        for act in text_acts:
                            act["page_number"] = page_num
                            act["source"] = "text"
                        all_acts.extend(text_acts)
                    
                    logger.info(f"Processed page {page_num}")
                    
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            pages_content.append({"error": str(e)})
        
        # Remove duplicates based on name
        unique_acts = []
        seen_names = set()
        
        for act in all_acts:
            if act["name"] not in seen_names:
                unique_acts.append(act)
                seen_names.add(act["name"])
        
        return pages_content, unique_acts
    
    def calculate_statistics(self, pages: List[Dict], acts: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from the extracted data"""
        total_text = " ".join([page.get("text", "") for page in pages])
        
        # Count different types of legal documents
        document_types = {
            "acts": len([act for act in acts if "act" in act["name"].lower()]),
            "codes": len([act for act in acts if "code" in act["name"].lower()]),
            "rules": len([act for act in acts if "rules" in act["name"].lower()]),
            "regulations": len([act for act in acts if "regulation" in act["name"].lower()]),
            "laws": len([act for act in acts if "law" in act["name"].lower()])
        }
        
        # Extract years from acts
        years = [act["year"] for act in acts if act.get("year")]
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        # Get year range
        int_years = [int(year) for year in years if year.isdigit()]
        
        return {
            "total_acts": len(acts),
            "document_types": document_types,
            "year_distribution": year_counts,
            "year_range": {
                "earliest": min(int_years) if int_years else None,
                "latest": max(int_years) if int_years else None
            },
            "total_characters": len(total_text),
            "total_words": len(total_text.split()),
            "pages_processed": len(pages)
        }
    
    def scrape_pdf(self) -> Dict[str, Any]:
        """Main method to scrape the PDF and return structured data"""
        logger.info(f"Starting to scrape PDF: {self.pdf_path}")
        
        # Extract metadata
        self.extracted_data["metadata"] = self.extract_metadata()
        
        # Extract content and acts
        pages, acts = self.extract_content_and_acts()
        self.extracted_data["raw_pages"] = pages
        self.extracted_data["acts"] = acts
        
        # Calculate statistics
        self.extracted_data["statistics"] = self.calculate_statistics(pages, acts)
        
        logger.info("PDF scraping completed successfully")
        return self.extracted_data
    
    def save_to_json(self, output_path: str = None):
        """Save the extracted data to a JSON file"""
        if output_path is None:
            output_path = self.pdf_path.parent / f"{self.pdf_path.stem}_structured_data.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Data saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return None
    
    def save_acts_only(self, output_path: str = None):
        """Save only the acts data to a separate JSON file"""
        if output_path is None:
            output_path = self.pdf_path.parent / f"{self.pdf_path.stem}_acts_only.json"
        
        acts_data = {
            "total_acts": len(self.extracted_data["acts"]),
            "extraction_date": str(Path(__file__).stat().st_mtime),
            "source_file": str(self.pdf_path),
            "acts": self.extracted_data["acts"]
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(acts_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Acts-only data saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving acts-only JSON: {e}")
            return None

def main():
    # Path to the PDF file
    pdf_path = "page/actsList.pdf"
    
    # Create scraper instance
    scraper = ImprovedPDFScraper(pdf_path)
    
    # Scrape the PDF
    data = scraper.scrape_pdf()
    
    # Save complete data to JSON
    full_output_file = scraper.save_to_json()
    
    # Save acts-only data
    acts_output_file = scraper.save_acts_only()
    
    # Print summary
    print("\n" + "="*60)
    print("PDF SCRAPING SUMMARY")
    print("="*60)
    print(f"PDF File: {pdf_path}")
    print(f"Total Pages: {data['metadata'].get('total_pages', 'Unknown')}")
    print(f"Total Acts Found: {data['statistics']['total_acts']}")
    print(f"Full Data JSON: {full_output_file}")
    print(f"Acts Only JSON: {acts_output_file}")
    
    # Print sample acts
    if data['acts']:
        print(f"\n{'='*60}")
        print("SAMPLE ACTS (First 10)")
        print("="*60)
        for i, act in enumerate(data['acts'][:10]):
            print(f"{i+1:2d}. [{act.get('serial_number', 'N/A'):>3s}] {act['name']}")
            print(f"    Year: {act.get('year', 'N/A')}, Act No: {act.get('act_number', 'N/A')}")
            print(f"    Page: {act.get('page_number', 'N/A')}, Source: {act.get('source', 'N/A')}")
            print()
    
    # Print statistics
    print("="*60)
    print("STATISTICS")
    print("="*60)
    stats = data['statistics']
    print(f"Total Acts: {stats['total_acts']}")
    print(f"Year Range: {stats['year_range']['earliest']} - {stats['year_range']['latest']}")
    print(f"Total Characters: {stats.get('total_characters', 0):,}")
    print(f"Total Words: {stats.get('total_words', 0):,}")
    
    print(f"\nDocument Types:")
    for doc_type, count in stats.get('document_types', {}).items():
        print(f"  - {doc_type.title()}: {count}")
    
    print(f"\nTop Years by Act Count:")
    year_dist = stats.get('year_distribution', {})
    sorted_years = sorted(year_dist.items(), key=lambda x: x[1], reverse=True)
    for year, count in sorted_years[:10]:
        print(f"  - {year}: {count} acts")

if __name__ == "__main__":
    main()
