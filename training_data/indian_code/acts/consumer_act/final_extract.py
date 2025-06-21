from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import List, Dict, Any

class ConsumerActExtractor:
    """Comprehensive extractor for Consumer Protection Act sections"""
    
    def __init__(self, html_file: str = "data.html"):
        self.html_file = html_file
        self.soup = None
        self.sections = []
        
    def load_html(self):
        """Load and parse the HTML file"""
        try:
            with open(self.html_file, "r", encoding="utf-8") as file:
                self.soup = BeautifulSoup(file, "html.parser")
                return True
        except Exception as e:
            print(f"❌ Error loading HTML file: {e}")
            return False
    
    def extract_sections(self) -> List[Dict[str, Any]]:
        """Extract sections using multiple parsing strategies"""
        if not self.soup:
            if not self.load_html():
                return []
        
        sections = []
        
        # Strategy 1: Find all bold elements that might contain section numbers
        bold_elements = self.soup.find_all(['b', 'strong'])
        bold_elements.extend(self.soup.find_all('span', style=lambda x: x and 'font-weight' in str(x).lower()))
        
        for element in bold_elements:
            text = element.get_text(strip=True)
            
            # Check if this looks like a section number
            section_match = re.match(r'^(\d+\.\d+(?:-\d+[a-z]*)?[a-z]?)', text)
            if section_match:
                section_number = section_match.group(1)
                heading = text[len(section_number):].strip()
                
                # Extract content following this section
                content = self._extract_section_content(element)
                
                if content and len(content) > 30:
                    sections.append({
                        "section_number": section_number,
                        "heading": heading,
                        "content": content,
                        "method": "bold_element"
                    })
        
        # Strategy 2: Parse full text for section patterns
        full_text = self.soup.get_text()
        text_sections = self._extract_from_text(full_text)
        
        # Combine and deduplicate sections
        combined_sections = self._combine_sections(sections, text_sections)
        
        # Clean and sort sections
        final_sections = self._clean_and_sort_sections(combined_sections)
        
        self.sections = final_sections
        return final_sections
    
    def _extract_section_content(self, element) -> str:
        """Extract content following a section header element"""
        content_parts = []
        
        # Get parent and following siblings
        current = element.parent
        if current:
            # Look for content in the same paragraph
            para_text = current.get_text(strip=True)
            # Remove the section number part
            section_match = re.match(r'^\d+\.\d+(?:-\d+[a-z]*)?[a-z]?', para_text)
            if section_match:
                remaining_text = para_text[len(section_match.group(0)):].strip()
                if remaining_text:
                    content_parts.append(remaining_text)
            
            # Look for following content
            next_elem = current.find_next_sibling()
            while next_elem and len(content_parts) < 5:  # Limit to avoid too much content
                next_text = next_elem.get_text(strip=True)
                
                # Stop if we find another section
                if re.match(r'^\d+\.\d+(?:-\d+[a-z]*)?[a-z]?', next_text):
                    break
                
                if next_text and len(next_text) > 10:
                    content_parts.append(next_text)
                
                next_elem = next_elem.find_next_sibling()
        
        return " ".join(content_parts)
    
    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections by parsing the full text"""
        sections = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_section = None
        
        for line in lines:
            # Check if line starts with a section number
            section_match = re.match(r'^(\d+\.\d+(?:-\d+[a-z]*)?[a-z]?)\s*(.*)', line)
            
            if section_match:
                # Save previous section
                if current_section and current_section.get("content"):
                    sections.append({
                        "section_number": current_section["number"],
                        "heading": current_section["heading"],
                        "content": current_section["content"].strip(),
                        "method": "text_parsing"
                    })
                
                # Start new section
                current_section = {
                    "number": section_match.group(1),
                    "heading": section_match.group(2),
                    "content": ""
                }
            
            elif current_section and not re.match(r'^\d+\.\d+', line):
                # Add content to current section (skip if it's another section)
                if len(line) > 5:  # Ignore very short lines
                    current_section["content"] += line + " "
        
        # Add last section
        if current_section and current_section.get("content"):
            sections.append({
                "section_number": current_section["number"],
                "heading": current_section["heading"],
                "content": current_section["content"].strip(),
                "method": "text_parsing"
            })
        
        return sections
    
    def _combine_sections(self, sections1: List[Dict], sections2: List[Dict]) -> List[Dict]:
        """Combine sections from different extraction methods"""
        section_map = {}
        
        # Add sections from first method
        for section in sections1:
            section_num = section["section_number"]
            section_map[section_num] = section
        
        # Add or merge sections from second method
        for section in sections2:
            section_num = section["section_number"]
            if section_num not in section_map:
                section_map[section_num] = section
            else:
                # Keep the one with more content
                existing = section_map[section_num]
                if len(section["content"]) > len(existing["content"]):
                    section_map[section_num] = section
        
        return list(section_map.values())
    
    def _clean_and_sort_sections(self, sections: List[Dict]) -> List[Dict]:
        """Clean up and sort sections"""
        # Clean up content
        for section in sections:
            content = section["content"]
            # Remove extra whitespace
            content = re.sub(r'\s+', ' ', content)
            # Remove section number if it appears at the start of content
            content = re.sub(r'^' + re.escape(section["section_number"]) + r'\s*', '', content)
            section["content"] = content.strip()
        
        # Filter out sections with insufficient content
        sections = [s for s in sections if len(s["content"]) > 30]
        
        # Sort sections
        def sort_key(section):
            parts = section["section_number"].split('.')
            try:
                main = int(parts[0])
                sub = parts[1] if len(parts) > 1 else "0"
                
                if '-' in sub:
                    sub_parts = sub.split('-')
                    sub_num = int(sub_parts[0])
                    sub_letter = sub_parts[1] if len(sub_parts) > 1 else ""
                    return (main, sub_num, sub_letter)
                else:
                    try:
                        return (main, int(sub), "")
                    except:
                        return (main, 0, sub)
            except:
                return (0, 0, section["section_number"])
        
        try:
            sections.sort(key=sort_key)
        except:
            pass  # Keep original order if sorting fails
        
        return sections
    
    def analyze_sections(self) -> Dict[str, Any]:
        """Analyze the extracted sections"""
        if not self.sections:
            return {}
        
        analysis = {
            "total_sections": len(self.sections),
            "main_sections": 0,
            "subsections": 0,
            "section_types": {},
            "content_statistics": {
                "total_words": 0,
                "total_characters": 0,
                "average_section_length": 0
            }
        }
        
        total_words = 0
        total_chars = 0
        
        for section in self.sections:
            section_num = section["section_number"]
            
            # Count main sections vs subsections
            if '-' in section_num:
                analysis["subsections"] += 1
            else:
                analysis["main_sections"] += 1
            
            # Analyze content
            content = section["content"]
            words = len(content.split())
            chars = len(content)
            
            total_words += words
            total_chars += chars
            
            # Track section types
            if section["heading"]:
                heading_key = section["heading"][:20] + "..." if len(section["heading"]) > 20 else section["heading"]
                analysis["section_types"][heading_key] = analysis["section_types"].get(heading_key, 0) + 1
        
        analysis["content_statistics"]["total_words"] = total_words
        analysis["content_statistics"]["total_characters"] = total_chars
        analysis["content_statistics"]["average_section_length"] = total_words // len(self.sections) if self.sections else 0
        
        return analysis
    
    def save_to_json(self, filename: str = "consumer_act_final.json"):
        """Save extracted sections to JSON"""
        data = {
            "metadata": {
                "act_name": "Consumer Protection Act",
                "extraction_date": datetime.now().isoformat(),
                "source_file": self.html_file,
                "total_sections": len(self.sections),
                "extraction_method": "comprehensive_multi_strategy"
            },
            "sections": [
                {
                    "section_number": section["section_number"],
                    "heading": section["heading"],
                    "full_text": section["content"],
                    "extraction_method": section.get("method", "unknown")
                }
                for section in self.sections
            ],
            "analysis": self.analyze_sections()
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save simplified version
        simple_filename = filename.replace(".json", "_simple.json")
        simple_data = [
            {
                "section": section["section_number"],
                "heading": section["heading"],
                "text": section["content"]
            }
            for section in self.sections
        ]
        
        with open(simple_filename, "w", encoding="utf-8") as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)
        
        return filename, simple_filename
    
    def display_summary(self):
        """Display extraction summary"""
        analysis = self.analyze_sections()
        
        print(f"📊 EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Sections: {analysis['total_sections']}")
        print(f"Main Sections: {analysis['main_sections']}")
        print(f"Subsections: {analysis['subsections']}")
        print(f"Total Words: {analysis['content_statistics']['total_words']:,}")
        print(f"Total Characters: {analysis['content_statistics']['total_characters']:,}")
        print(f"Average Section Length: {analysis['content_statistics']['average_section_length']} words")
        
        if self.sections:
            print(f"Section Range: {self.sections[0]['section_number']} to {self.sections[-1]['section_number']}")
        
        print(f"\n📝 SAMPLE SECTIONS:")
        print(f"{'='*60}")
        for i, section in enumerate(self.sections[:5]):
            print(f"\n{i+1}. Section {section['section_number']}")
            if section['heading']:
                print(f"   Heading: {section['heading']}")
            print(f"   Content: {section['content'][:150]}...")

def main():
    """Main function"""
    print("="*80)
    print("CONSUMER PROTECTION ACT - COMPREHENSIVE SECTION EXTRACTOR")
    print("="*80)
    
    # Create extractor and extract sections
    extractor = ConsumerActExtractor()
    sections = extractor.extract_sections()
    
    if not sections:
        print("❌ No sections extracted!")
        return
    
    # Save to JSON files
    main_file, simple_file = extractor.save_to_json()
    
    print(f"✅ Sections extracted and saved!")
    print(f"   Main file: {main_file}")
    print(f"   Simple file: {simple_file}")
    
    # Display summary
    extractor.display_summary()

if __name__ == "__main__":
    main()
