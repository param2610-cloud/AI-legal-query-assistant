#!/usr/bin/env python3
"""
Extract and structure data from to_be_extract.txt into JSON format
Compatible with existing legal acts data structure
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any

def parse_acts_table(file_path: str) -> List[Dict[str, Any]]:
    """Parse the acts table from the text file"""
    acts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Skip header lines (first 2 lines)
        for line in lines[2:]:
            line = line.strip()
            if not line or line.startswith('|') == False:
                continue
            
            # Split by pipe and clean up
            parts = [part.strip() for part in line.split('|')]
            
            # Filter out empty parts and separators
            parts = [part for part in parts if part and not all(c in '-' for c in part)]
            
            if len(parts) >= 2:
                act_name = parts[0]
                year_info = parts[1]
                
                # Handle multiple years (e.g., "1986, 2019")
                years = []
                if ',' in year_info:
                    years = [y.strip() for y in year_info.split(',')]
                else:
                    years = [year_info.strip()]
                
                # Create act entry
                act_data = {
                    "name": act_name,
                    "years": years,
                    "primary_year": years[0] if years else None,
                    "status": "pending_extraction",
                    "categories": determine_categories(act_name)
                }
                
                acts.append(act_data)
    
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        return []
    
    return acts

def determine_categories(act_name: str) -> List[str]:
    """Determine categories based on act name"""
    categories = []
    name_lower = act_name.lower()
    
    # Family and Marriage Laws
    if any(keyword in name_lower for keyword in ['marriage', 'divorce', 'muslim', 'hindu', 'christian']):
        categories.append("Family Law")
    
    # Children and Youth
    if any(keyword in name_lower for keyword in ['children', 'juvenile', 'child', 'pocso']):
        categories.append("Child Protection")
    
    # Women Protection
    if any(keyword in name_lower for keyword in ['women', 'domestic violence']):
        categories.append("Women Protection")
    
    # Labor and Employment
    if any(keyword in name_lower for keyword in ['provident', 'gratuity', 'employees']):
        categories.append("Labor Law")
    
    # Rights and Information
    if any(keyword in name_lower for keyword in ['right', 'information', 'education']):
        categories.append("Fundamental Rights")
    
    # Criminal Law
    if any(keyword in name_lower for keyword in ['narcotic', 'drugs', 'offences']):
        categories.append("Criminal Law")
    
    # Traffic and Vehicles
    if any(keyword in name_lower for keyword in ['motor', 'vehicles']):
        categories.append("Traffic Law")
    
    # Contracts and Civil
    if any(keyword in name_lower for keyword in ['contract']):
        categories.append("Contract Law")
    
    # Default category if none matched
    if not categories:
        categories.append("General Law")
    
    return categories

def create_structured_json(acts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create structured JSON similar to existing legal acts"""
    
    # Count statistics
    total_acts = len(acts)
    categories_count = {}
    years_count = {}
    
    for act in acts:
        # Count categories
        for category in act["categories"]:
            categories_count[category] = categories_count.get(category, 0) + 1
        
        # Count years
        primary_year = act["primary_year"]
        if primary_year:
            years_count[primary_year] = years_count.get(primary_year, 0) + 1
    
    structured_data = {
        "metadata": {
            "dataset_name": "Pending Legal Acts for Extraction",
            "extraction_date": datetime.now().isoformat(),
            "source_file": "to_be_extract.txt",
            "total_acts": total_acts,
            "status": "pending_extraction",
            "description": "List of Indian legal acts that need to be processed and added to the AI Legal Assistant database"
        },
        "statistics": {
            "total_acts": total_acts,
            "categories_distribution": categories_count,
            "years_distribution": years_count,
            "oldest_act": min(years_count.keys()) if years_count else None,
            "newest_act": max(years_count.keys()) if years_count else None
        },
        "acts": acts,
        "extraction_plan": {
            "high_priority": [
                "Right to Information Act",
                "Motor Vehicles Act",
                "Protection of Women from Domestic Violence Act",
                "Protection of Children from Sexual Offences (POCSO) Act"
            ],
            "medium_priority": [
                "Hindu Marriage Act",
                "Indian Contract Act",
                "Narcotic Drugs and Psychotropic Substances Act"
            ],
            "low_priority": [
                "Payment of Gratuity Act",
                "Employees' Provident Funds and Miscellaneous Provisions Act"
            ]
        }
    }
    
    return structured_data

def create_simple_format(acts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create simplified format for easy processing"""
    simple_acts = []
    
    for act in acts:
        simple_act = {
            "act_name": act["name"],
            "year": act["primary_year"],
            "all_years": act["years"],
            "categories": act["categories"],
            "priority": determine_priority(act["name"]),
            "extraction_status": "pending"
        }
        simple_acts.append(simple_act)
    
    return simple_acts

def determine_priority(act_name: str) -> str:
    """Determine extraction priority"""
    high_priority_keywords = ['right to information', 'motor vehicles', 'domestic violence', 'pocso', 'children']
    medium_priority_keywords = ['marriage', 'contract', 'narcotic', 'education']
    
    name_lower = act_name.lower()
    
    if any(keyword in name_lower for keyword in high_priority_keywords):
        return "high"
    elif any(keyword in name_lower for keyword in medium_priority_keywords):
        return "medium"
    else:
        return "low"

def main():
    """Main extraction function"""
    print("=" * 80)
    print("PENDING LEGAL ACTS - DATA EXTRACTION")
    print("=" * 80)
    
    input_file = "to_be_extract.txt"
    
    # Parse the acts from the text file
    print(f"📖 Reading acts from {input_file}...")
    acts = parse_acts_table(input_file)
    
    if not acts:
        print("❌ No acts found in the file!")
        return
    
    print(f"✅ Found {len(acts)} acts to process")
    
    # Create structured JSON
    print("🔄 Creating structured JSON...")
    structured_data = create_structured_json(acts)
    
    # Create simplified format
    print("🔄 Creating simplified format...")
    simple_data = create_simple_format(acts)
    
    # Save structured JSON
    structured_file = "pending_acts_structured.json"
    with open(structured_file, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    # Save simplified JSON
    simple_file = "pending_acts_simple.json"
    with open(simple_file, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    # Save acts-only format (compatible with existing structure)
    acts_only_file = "pending_acts_only.json"
    acts_only = [
        {
            "act": act["name"],
            "year": act["primary_year"],
            "years": act["years"],
            "categories": act["categories"]
        }
        for act in acts
    ]
    
    with open(acts_only_file, 'w', encoding='utf-8') as f:
        json.dump(acts_only, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Files created successfully:")
    print(f"   📄 {structured_file} - Complete structured data")
    print(f"   📄 {simple_file} - Simplified format")
    print(f"   📄 {acts_only_file} - Acts-only format")
    
    # Display summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    
    stats = structured_data["statistics"]
    print(f"Total Acts: {stats['total_acts']}")
    print(f"Year Range: {stats['oldest_act']} - {stats['newest_act']}")
    
    print("\n📊 Category Distribution:")
    for category, count in sorted(stats['categories_distribution'].items()):
        print(f"   {category}: {count} acts")
    
    print("\n🎯 Priority Distribution:")
    priority_counts = {"high": 0, "medium": 0, "low": 0}
    for act in simple_data:
        priority_counts[act["priority"]] += 1
    
    for priority, count in priority_counts.items():
        print(f"   {priority.title()} Priority: {count} acts")
    
    print("\n📋 Sample Acts:")
    for i, act in enumerate(acts[:5]):
        print(f"   {i+1}. {act['name']} ({act['primary_year']}) - {', '.join(act['categories'])}")
    
    print("\n✨ Next Steps:")
    print("   1. Review the generated JSON files")
    print("   2. Start with high-priority acts")
    print("   3. Create individual extraction scripts for each act")
    print("   4. Integrate extracted data into the RAG system")

if __name__ == "__main__":
    main()
