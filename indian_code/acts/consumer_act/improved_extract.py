from bs4 import BeautifulSoup
import json
import re
from datetime import datetime

def extract_sections_from_html():
    """Extract sections from Consumer Protection Act HTML with improved parsing"""
    
    # Load HTML content
    with open("data.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    
    # Initialize data structure for JSON storage
    consumer_act_data = {
        "metadata": {
            "act_name": "Consumer Protection Act",
            "extraction_date": datetime.now().isoformat(),
            "source_file": "data.html",
            "total_sections": 0
        },
        "sections": []
    }
    
    # Extract all text content from paragraphs
    all_paragraphs = soup.find_all("p")
    text_content = []
    
    for para in all_paragraphs:
        text = para.get_text(strip=True)
        if text and len(text) > 10:  # Filter out very short or empty paragraphs
            text_content.append(text)
    
    # Join all text and split by lines for processing
    full_text = "\n".join(text_content)
    
    # Improved regex patterns for section identification
    # Pattern 1: Bold sections like <b><span>1.1</span></b>
    # Pattern 2: Standalone section numbers
    section_patterns = [
        r'(\d+\.\d+(?:-\d+[a-z]*)?)\s+(.*?)(?=\d+\.\d+(?:-\d+[a-z]*)?|$)',  # Main pattern
        r'(\d+\.\d+(?:-\d+[a-z]*)?[a-z]?)\s*(.+?)(?=(?:\d+\.\d+)|(?:$))',    # Alternative pattern
    ]
    
    # Find sections using direct HTML parsing for better accuracy
    sections = []
    
    # Look for bold spans containing section numbers
    bold_spans = soup.find_all('span', style=lambda x: x and 'font-weight' in str(x).lower())
    bold_spans.extend(soup.find_all('b'))
    
    for element in bold_spans:
        text = element.get_text(strip=True)
        
        # Check if this looks like a section number
        section_match = re.match(r'^(\d+\.\d+(?:-\d+[a-z]*)?[a-z]?)', text)
        if section_match:
            section_number = section_match.group(1)
            
            # Get the heading (rest of the bold text)
            heading = text[len(section_number):].strip()
            
            # Get the following content until next section
            content = ""
            current = element.parent
            
            # Navigate to get the section content
            if current:
                # Look for the next paragraph or content after this element
                next_elem = current.find_next_sibling()
                content_parts = []
                
                # Collect content until we find another section
                while next_elem:
                    next_text = next_elem.get_text(strip=True)
                    
                    # Stop if we find another section number
                    if re.match(r'^\d+\.\d+(?:-\d+[a-z]*)?[a-z]?', next_text):
                        break
                    
                    if next_text and len(next_text) > 5:
                        content_parts.append(next_text)
                    
                    next_elem = next_elem.find_next_sibling()
                    
                    # Limit to avoid infinite loops
                    if len(content_parts) > 10:
                        break
                
                content = " ".join(content_parts)
            
            # Add to sections if we have meaningful content
            if content and len(content) > 20:
                sections.append({
                    "section_number": section_number,
                    "heading": heading,
                    "full_text": content
                })
    
    # Alternative approach: Parse by looking for section patterns in the text
    lines = full_text.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if line starts with a section number
        section_match = re.match(r'^(\d+\.\d+(?:-\d+[a-z]*)?[a-z]?)\s*(.*)', line)
        
        if section_match:
            # Save previous section if exists
            if current_section and current_section.get("content"):
                sections.append({
                    "section_number": current_section["number"],
                    "heading": current_section["heading"],
                    "full_text": current_section["content"]
                })
            
            # Start new section
            section_num = section_match.group(1)
            section_heading = section_match.group(2)
            
            current_section = {
                "number": section_num,
                "heading": section_heading,
                "content": ""
            }
        
        elif current_section and line and not re.match(r'^\d+\.\d+', line):
            # Add content to current section
            current_section["content"] += line + " "
    
    # Add the last section
    if current_section and current_section.get("content"):
        sections.append({
            "section_number": current_section["number"],
            "heading": current_section["heading"],
            "full_text": current_section["content"].strip()
        })
    
    # Remove duplicates and clean up
    unique_sections = []
    seen_numbers = set()
    
    for section in sections:
        section_num = section["section_number"]
        if section_num not in seen_numbers and len(section["full_text"]) > 30:
            unique_sections.append(section)
            seen_numbers.add(section_num)
    
    # Sort sections by section number
    def sort_key(section):
        parts = section["section_number"].split('.')
        main = int(parts[0])
        sub = parts[1] if len(parts) > 1 else "0"
        
        # Handle subsections like 1.2-1a
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
    
    try:
        unique_sections.sort(key=sort_key)
    except:
        pass  # If sorting fails, keep original order
    
    # Update the data structure
    consumer_act_data["sections"] = unique_sections
    consumer_act_data["metadata"]["total_sections"] = len(unique_sections)
    
    return consumer_act_data

def save_extracted_data(data):
    """Save the extracted data to JSON files"""
    
    # Save complete data to JSON
    with open("consumer_act_improved.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)
    
    print("✅ Extracted and stored sections to consumer_act_improved.json")
    print(f"📊 Total sections extracted: {data['metadata']['total_sections']}")
    
    # Save simplified version with just sections
    sections_only = [
        {
            "section": section["section_number"],
            "heading": section["heading"],
            "text": section["full_text"]
        }
        for section in data["sections"]
    ]
    
    with open("consumer_act_sections_improved.json", "w", encoding="utf-8") as json_file:
        json.dump(sections_only, json_file, indent=2, ensure_ascii=False)
    
    print("✅ Also saved simplified sections to consumer_act_sections_improved.json")
    
    return sections_only

def display_sample_sections(sections, count=10):
    """Display a sample of extracted sections"""
    
    print(f"\n{'='*80}")
    print(f"SAMPLE SECTIONS (First {count})")
    print(f"{'='*80}")
    
    for i, section in enumerate(sections[:count]):
        print(f"\n{i+1:2d}. Section {section['section']}")
        print(f"    Heading: {section['heading']}")
        print(f"    Text: {section['text'][:200]}...")
        print("-" * 60)

def main():
    """Main function to extract and display sections"""
    
    print("="*80)
    print("CONSUMER PROTECTION ACT - SECTION EXTRACTOR (IMPROVED)")
    print("="*80)
    
    try:
        # Extract sections
        data = extract_sections_from_html()
        
        # Save data
        sections = save_extracted_data(data)
        
        # Display sample
        display_sample_sections(sections)
        
        # Show statistics
        print(f"\n{'='*80}")
        print("EXTRACTION STATISTICS")
        print(f"{'='*80}")
        print(f"Total sections found: {len(sections)}")
        
        # Count different section types
        main_sections = len([s for s in sections if '-' not in s['section']])
        subsections = len([s for s in sections if '-' in s['section']])
        
        print(f"Main sections: {main_sections}")
        print(f"Subsections: {subsections}")
        
        # Show section number range
        section_numbers = [s['section'] for s in sections]
        print(f"Section range: {section_numbers[0]} to {section_numbers[-1]}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
