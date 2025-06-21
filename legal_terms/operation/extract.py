#!/usr/bin/env python3
"""
Legal Terms Extractor for AI Legal Query Assistant

This script extracts legal terms and definitions from the HTML file and creates 
a structured JSON that will be useful for AI agent operations.
"""

import re
import json
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import os

class LegalTermsExtractor:
    """
    Comprehensive extractor for legal terms and definitions from HTML
    """
    
    def __init__(self, html_file: str = "data.html"):
        self.html_file = html_file
        self.soup = None
        self.terms = []
        
    def load_html(self) -> bool:
        """Load and parse the HTML file"""
        try:
            if not os.path.exists(self.html_file):
                print(f"❌ Error: HTML file '{self.html_file}' not found")
                return False
                
            with open(self.html_file, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            self.soup = BeautifulSoup(html_content, 'html.parser')
            print(f"✅ Successfully loaded HTML file: {self.html_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading HTML file: {str(e)}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
            
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove HTML artifacts
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        
        # Clean up formatting
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_term_and_definition(self, paragraph_text: str) -> Optional[Dict[str, Any]]:
        """Extract term and definition from a paragraph"""
        # Pattern to match terms with various formats
        # The HTML has format: "Term- Definition" or "Term: Definition"
        
        # First, try to match the bold term pattern
        patterns = [
            r'^([A-Za-z][^-–:]*?)[-–:]\s*(.+)$',  # Term- or Term: format
            r'^([A-Za-z][^-–:]*?)\s*[-–]\s*(.+)$',  # Term - format with spaces
        ]
        
        for pattern in patterns:
            match = re.match(pattern, paragraph_text)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                # Clean the term
                term = re.sub(r'\s+', ' ', term)
                term = term.strip()
                
                # Remove common trailing characters from term
                term = re.sub(r'[^\w\s\(\)]+$', '', term)
                term = term.strip()
                
                # Validate term and definition
                if len(term) < 2 or len(definition) < 5:
                    continue
                    
                # Clean definition
                definition = self.clean_definition(definition)
                
                if len(definition) < 10:
                    continue
                
                return {
                    "term": term,
                    "definition": definition,
                    "length": len(definition),
                    "word_count": len(definition.split())
                }
        
        return None
    
    def clean_definition(self, definition: str) -> str:
        """Clean and process definition text"""
        # Remove wiki links and references
        definition = re.sub(r'Check the wiki page.*$', '', definition, flags=re.IGNORECASE)
        definition = re.sub(r'See http.*$', '', definition, flags=re.IGNORECASE)
        definition = re.sub(r'http[s]?://[^\s]+', '', definition, flags=re.IGNORECASE)
        
        # Remove reference patterns
        definition = re.sub(r'As per the \d+.*?report:\s*', '', definition, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        definition = re.sub(r'\s+', ' ', definition)
        definition = definition.strip()
        
        # Remove trailing periods or other punctuation if followed by nothing
        definition = re.sub(r'[.,:;]+$', '.', definition)
        
        return definition
    
    def categorize_term(self, term: str, definition: str) -> List[str]:
        """Categorize legal terms based on content"""
        categories = []
        
        term_lower = term.lower()
        definition_lower = definition.lower()
        
        # Criminal Law Terms
        criminal_keywords = [
            'crime', 'criminal', 'accused', 'arrest', 'charge', 'conviction', 
            'bail', 'custody', 'investigation', 'prosecution', 'murder', 
            'theft', 'assault', 'rape', 'homicide', 'robbery', 'arson', 'fir'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in criminal_keywords):
            categories.append("Criminal Law")
        
        # Civil Law Terms
        civil_keywords = [
            'civil', 'plaintiff', 'defendant', 'suit', 'decree', 'judgment', 
            'damages', 'contract', 'tort', 'negligence', 'liability'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in civil_keywords):
            categories.append("Civil Law")
        
        # Court Procedure Terms
        procedure_keywords = [
            'court', 'procedure', 'hearing', 'trial', 'appeal', 'jurisdiction', 
            'summons', 'notice', 'pleading', 'evidence', 'witness', 'testimony'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in procedure_keywords):
            categories.append("Court Procedure")
        
        # Legal Documents
        document_keywords = [
            'document', 'affidavit', 'petition', 'application', 'plaint', 
            'written statement', 'order', 'warrant'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in document_keywords):
            categories.append("Legal Documents")
        
        # Constitutional Law
        constitutional_keywords = [
            'constitution', 'fundamental rights', 'writ', 'mandamus', 
            'habeas corpus', 'certiorari', 'prohibition', 'quo warranto'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in constitutional_keywords):
            categories.append("Constitutional Law")
        
        # Legal Professionals
        professional_keywords = [
            'judge', 'advocate', 'lawyer', 'magistrate', 'prosecutor'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in professional_keywords):
            categories.append("Legal Professionals")
        
        # Alternative Dispute Resolution
        adr_keywords = [
            'mediation', 'arbitration', 'lok adalat', 'settlement', 'compromise'
        ]
        if any(keyword in term_lower or keyword in definition_lower for keyword in adr_keywords):
            categories.append("Alternative Dispute Resolution")
        
        return categories if categories else ["General Legal"]
    
    def extract_all_terms(self) -> List[Dict[str, Any]]:
        """Extract all legal terms from the HTML"""
        if not self.soup:
            print("❌ HTML not loaded. Call load_html() first.")
            return []
        
        terms = []
        
        # Find all paragraphs containing legal terms
        paragraphs = self.soup.find_all('p')
        print(f"🔍 Found {len(paragraphs)} paragraphs to process")
        
        for i, para in enumerate(paragraphs):
            # Get text content
            text = para.get_text()
            text = self.clean_text(text)
            
            if not text or len(text) < 20:
                continue
            
            # Debug: print first few paragraphs
            if i < 5:
                print(f"   Processing paragraph {i+1}: {text[:100]}...")
            
            # Extract term and definition
            term_data = self.extract_term_and_definition(text)
            
            if term_data:
                # Add categories
                categories = self.categorize_term(term_data["term"], term_data["definition"])
                term_data["categories"] = categories
                
                # Add additional metadata
                term_data["source"] = "Legal Terms HTML"
                term_data["extraction_date"] = datetime.now().isoformat()
                
                # Create alternate representations for better AI matching
                term_variants = self.generate_term_variants(term_data["term"])
                term_data["variants"] = term_variants
                
                # Add keyword extraction for better searchability
                keywords = self.extract_keywords(term_data["definition"])
                term_data["keywords"] = keywords
                
                terms.append(term_data)
                
                if len(terms) <= 5:  # Show first few extracted terms
                    print(f"   ✅ Extracted: {term_data['term']}")
        
        self.terms = terms
        print(f"✅ Extracted {len(terms)} legal terms")
        return terms
    
    def generate_term_variants(self, term: str) -> List[str]:
        """Generate variants of the term for better matching"""
        variants = [term]
        
        # Add lowercase version
        if term.lower() != term:
            variants.append(term.lower())
        
        # Add title case version
        if term.title() != term:
            variants.append(term.title())
        
        # Add version without special characters
        clean_term = re.sub(r'[^\w\s]', '', term)
        if clean_term != term and clean_term not in variants:
            variants.append(clean_term)
        
        # Add hyphenated versions
        if ' ' in term:
            hyphenated = term.replace(' ', '-')
            if hyphenated not in variants:
                variants.append(hyphenated)
        
        return list(set(variants))  # Remove duplicates
    
    def extract_keywords(self, definition: str) -> List[str]:
        """Extract important keywords from definition"""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'his', 'her', 'they', 'them',
            'their', 'this', 'can', 'may', 'have', 'had', 'been', 'which',
            'who', 'what', 'where', 'when', 'why', 'how', 'such', 'should',
            'would', 'could', 'shall', 'must', 'person', 'case', 'court'
        }
        
        # Extract words, filter and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', definition.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency and take most common
        from collections import Counter
        word_counts = Counter(keywords)
        
        # Return top keywords
        return [word for word, count in word_counts.most_common(10)]
    
    def create_ai_optimized_structure(self) -> Dict[str, Any]:
        """Create AI-optimized data structure"""
        if not self.terms:
            return {}
        
        # Create multiple access patterns for AI
        structure = {
            "metadata": {
                "source": "Legal Terms Dictionary",
                "extraction_date": datetime.now().isoformat(),
                "total_terms": len(self.terms),
                "categories": list(set(cat for term in self.terms for cat in term["categories"])),
                "version": "1.0"
            },
            
            # Main terms list
            "terms": self.terms,
            
            # Terms indexed by first letter for quick lookup
            "terms_by_letter": {},
            
            # Terms grouped by category
            "terms_by_category": {},
            
            # Term variants mapping for fuzzy matching
            "term_variants": {},
            
            # Keywords index for semantic search
            "keywords_index": {},
            
            # Quick lookup dictionary
            "term_definitions": {}
        }
        
        # Build indexes
        for term_data in self.terms:
            term = term_data["term"]
            definition = term_data["definition"]
            
            # Index by first letter
            first_letter = term[0].upper()
            if first_letter not in structure["terms_by_letter"]:
                structure["terms_by_letter"][first_letter] = []
            structure["terms_by_letter"][first_letter].append(term_data)
            
            # Index by category
            for category in term_data["categories"]:
                if category not in structure["terms_by_category"]:
                    structure["terms_by_category"][category] = []
                structure["terms_by_category"][category].append(term_data)
            
            # Build variants mapping
            for variant in term_data["variants"]:
                structure["term_variants"][variant.lower()] = term
            
            # Build keywords index
            for keyword in term_data["keywords"]:
                if keyword not in structure["keywords_index"]:
                    structure["keywords_index"][keyword] = []
                structure["keywords_index"][keyword].append(term)
            
            # Quick lookup
            structure["term_definitions"][term.lower()] = definition
        
        return structure
    
    def save_to_files(self) -> Dict[str, str]:
        """Save extracted terms to multiple JSON files"""
        if not self.terms:
            print("❌ No terms extracted. Run extract_all_terms() first.")
            return {}
        
        files_created = {}
        
        # 1. Complete structured data for AI operations
        ai_structure = self.create_ai_optimized_structure()
        ai_file = "legal_terms_structured.json"
        with open(ai_file, 'w', encoding='utf-8') as f:
            json.dump(ai_structure, f, indent=2, ensure_ascii=False)
        files_created["ai_structured"] = ai_file
        
        # 2. Simple terms list for basic operations
        simple_terms = [
            {
                "term": term["term"],
                "definition": term["definition"],
                "categories": term["categories"]
            }
            for term in self.terms
        ]
        simple_file = "legal_terms_simple.json"
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_terms, f, indent=2, ensure_ascii=False)
        files_created["simple"] = simple_file
        
        # 3. Dictionary format for quick lookups
        dictionary = {term["term"]: term["definition"] for term in self.terms}
        dict_file = "legal_terms_dictionary.json"
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
        files_created["dictionary"] = dict_file
        
        # 4. Category-wise breakdown
        categories_data = {}
        for term in self.terms:
            for category in term["categories"]:
                if category not in categories_data:
                    categories_data[category] = []
                categories_data[category].append({
                    "term": term["term"],
                    "definition": term["definition"]
                })
        
        categories_file = "legal_terms_by_category.json"
        with open(categories_file, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, indent=2, ensure_ascii=False)
        files_created["categories"] = categories_file
        
        # 5. Search-optimized format with keywords
        search_data = {
            "terms": [
                {
                    "term": term["term"],
                    "definition": term["definition"],
                    "keywords": term["keywords"],
                    "variants": term["variants"],
                    "categories": term["categories"]
                }
                for term in self.terms
            ],
            "keyword_index": ai_structure["keywords_index"],
            "variant_index": ai_structure["term_variants"]
        }
        
        search_file = "legal_terms_search_optimized.json"
        with open(search_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, indent=2, ensure_ascii=False)
        files_created["search_optimized"] = search_file
        
        return files_created
    
    def display_analysis(self):
        """Display analysis of extracted terms"""
        if not self.terms:
            print("❌ No terms to analyze")
            return
        
        print("=" * 80)
        print("LEGAL TERMS EXTRACTION ANALYSIS")
        print("=" * 80)
        
        # Basic statistics
        total_terms = len(self.terms)
        total_definitions = sum(1 for term in self.terms if term["definition"])
        avg_definition_length = sum(term["word_count"] for term in self.terms) / total_terms
        
        print(f"📊 EXTRACTION STATISTICS:")
        print(f"   Total Terms Extracted: {total_terms}")
        print(f"   Terms with Definitions: {total_definitions}")
        print(f"   Average Definition Length: {avg_definition_length:.1f} words")
        
        # Category breakdown
        category_counts = {}
        for term in self.terms:
            for category in term["categories"]:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\n🏷️  CATEGORY BREAKDOWN:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {category}: {count} terms")
        
        # Sample terms
        print(f"\n📝 SAMPLE TERMS:")
        for i, term in enumerate(self.terms[:5]):
            print(f"   {i+1}. {term['term']}")
            print(f"      Categories: {', '.join(term['categories'])}")
            print(f"      Definition: {term['definition'][:100]}...")
            print()

def main():
    """Main execution function"""
    print("=" * 80)
    print("LEGAL TERMS EXTRACTOR FOR AI LEGAL QUERY ASSISTANT")
    print("=" * 80)
    
    # Initialize extractor
    extractor = LegalTermsExtractor("data.html")
    
    # Load HTML
    if not extractor.load_html():
        return
    
    # Extract terms
    print("\n🔍 Extracting legal terms...")
    terms = extractor.extract_all_terms()
    
    if not terms:
        print("❌ No terms extracted. Please check the HTML file format.")
        return
    
    # Save to files
    print("\n💾 Saving extracted terms to files...")
    files_created = extractor.save_to_files()
    
    print("\n✅ Files created:")
    for file_type, filename in files_created.items():
        print(f"   {file_type}: {filename}")
    
    # Display analysis
    extractor.display_analysis()
    
    print("\n🎯 AI AGENT INTEGRATION NOTES:")
    print("   • Use 'legal_terms_structured.json' for comprehensive AI operations")
    print("   • Use 'legal_terms_search_optimized.json' for semantic search")
    print("   • Use 'legal_terms_dictionary.json' for quick term lookups")
    print("   • Use 'legal_terms_by_category.json' for category-specific queries")
    
    print(f"\n✅ Legal terms extraction completed successfully!")
    print(f"   📁 Output files are ready for AI agent integration")

if __name__ == "__main__":
    main()
