import pdfplumber
import re
import json
from datetime import datetime

# Path to Dowry Prohibition Act PDF
pdf_path = "dowry_prohibition.pdf"

print("📖 Starting Dowry Prohibition Act PDF extraction...")

# Initialize data structure for JSON storage
dowry_act_data = {
    "metadata": {
        "act_name": "Dowry Prohibition Act",
        "extraction_date": datetime.now().isoformat(),
        "source_file": "dowry_prohibition.pdf",
        "total_sections": 0
    },
    "sections": []
}

sections = {}

try:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        print(f"📄 Processing {len(pdf.pages)} pages...")
        
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
            print(f"   Page {page_num} processed")

    # Clean the text and remove excessive whitespace
    full_text = re.sub(r'\n+', '\n', full_text)
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # Regular expression to match each section start (e.g., "1. Short title, ...")
    section_pattern = re.compile(r'(?P<section_num>\d+[A-Z]?)\.\s*(?P<section_title>[^\.]+?)(?=\s*\(?\d|\s*[A-Z]{2,}|\s*$)', re.MULTILINE)

    # Split based on sections
    matches = list(section_pattern.finditer(full_text))
    print(f"🔍 Found {len(matches)} sections using regex pattern")

    # Alternative approach: split by section numbers and extract content
    section_splits = re.split(r'\b(\d+[A-Z]?)\.\s*', full_text)
    
    # Process splits to create sections
    temp_sections = {}
    for i in range(1, len(section_splits), 2):
        if i + 1 < len(section_splits):
            section_num = section_splits[i]
            section_content = section_splits[i + 1].strip()
            
            # Extract title (first line/sentence)
            lines = section_content.split('\n')
            if lines:
                title = lines[0].strip()
                # If title is too long, take first part
                if len(title) > 100:
                    title = title[:100] + "..."
                
                # Content is everything after the title
                content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else section_content
                
                temp_sections[section_num] = {
                    "title": title,
                    "content": content
                }

    print(f"🔄 Alternative method found {len(temp_sections)} sections")

    # Use the temp_sections if it has more meaningful content
    if temp_sections and len(temp_sections) > 5:
        sections = temp_sections
        print(f"✅ Using alternative extraction method, found {len(sections)} sections")
    else:
        # Fallback to original method
        for i, match in enumerate(matches):
            section_id = match.group("section_num")
            title = match.group("section_title").strip()

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

            content = full_text[start:end].strip()
            
            # Store in both formats
            sections[section_id] = {
                "title": title,
                "content": content
            }
        print(f"✅ Using regex extraction method, found {len(sections)} sections")

    # Convert sections dict to structured format
    for section_id, section_info in sections.items():
        section_data = {
            "section_number": section_id,
            "heading": section_info["title"],
            "full_text": section_info["content"]
        }
        dowry_act_data["sections"].append(section_data)

    # Update metadata with total sections count
    dowry_act_data["metadata"]["total_sections"] = len(dowry_act_data["sections"])

    # Output as JSON (original format)
    with open("dowry_sections.json", "w", encoding='utf-8') as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    # Output structured format (similar to consumer act)
    with open("dowry_act_structured.json", "w", encoding='utf-8') as f:
        json.dump(dowry_act_data, f, indent=2, ensure_ascii=False)

    # Also save a simplified version with just sections for easy access
    sections_only = [
        {
            "section": section["section_number"],
            "heading": section["heading"],
            "text": section["full_text"]
        }
        for section in dowry_act_data["sections"]
    ]

    with open("dowry_act_sections_only.json", "w", encoding='utf-8') as f:
        json.dump(sections_only, f, indent=2, ensure_ascii=False)

    print("✅ Dowry Prohibition Act sections saved to dowry_sections.json")
    print("✅ Structured format saved to dowry_act_structured.json")
    print("✅ Simplified sections saved to dowry_act_sections_only.json")
    print(f"📊 Total sections extracted: {dowry_act_data['metadata']['total_sections']}")

    # Display a preview of the first few sections
    if sections:
        print("\n📋 Preview of extracted sections:")
        for i, (section_id, section_info) in enumerate(list(sections.items())[:3]):
            print(f"   Section {section_id}: {section_info['title'][:60]}{'...' if len(section_info['title']) > 60 else ''}")

except Exception as e:
    print(f"❌ Error extracting PDF: {str(e)}")
    print("Make sure pdfplumber is installed: pip install pdfplumber")
    print("Also ensure that dowry_prohibition.pdf exists in the current directory")