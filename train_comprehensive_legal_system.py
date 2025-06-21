#!/usr/bin/env python3
"""
Comprehensive Legal System Training Script
==========================================

This script trains the AI Legal Assistant with all available legal data including:
- Indian Code Acts (Child Labour, Civil Procedure, Consumer Protection, etc.)
- Legal Terms and Definitions
- Legal Procedures (Civil, Criminal, Consumer Complaints, etc.)
- Legal Forms and Templates
- Jurisdictional Information
- Court Hierarchy and Geographic Data

Usage:
    python train_comprehensive_legal_system.py
    python train_comprehensive_legal_system.py --rebuild-all
    python train_comprehensive_legal_system.py --acts-only
    python train_comprehensive_legal_system.py --terms-only
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Please run: pip install chromadb sentence-transformers pandas")
    DEPENDENCIES_AVAILABLE = False

class ComprehensiveLegalTrainer:
    """Main trainer class for all legal data"""
    
    def __init__(self, base_dir: str = "training_data"):
        self.base_dir = Path(base_dir)
        self.chroma_dir = Path("chroma_db_comprehensive")
        self.embedding_model = None
        self.chroma_client = None
        self.collections = {}
        
        # Data source directories
        self.data_sources = {
            'indian_acts': self.base_dir / "indian_code" / "acts",
            'legal_terms': self.base_dir / "legal_terms",
            'procedures': self.base_dir / "procedure",
            'forms_templates': self.base_dir / "legal_forms_templates",
            'jurisdictional': self.base_dir / "jurisdictional_info",
            'case_law': self.base_dir / "case_law",
            'geographical': self.base_dir / "geographical_jurisdiction",
            'advocate_data': self.base_dir / "advocate_data",
            'emergency_data': self.base_dir / "emergency_data",
            'fees': self.base_dir / "Fees"
        }
        
    def initialize_system(self):
        """Initialize the training system"""
        logger.info("🚀 Initializing Comprehensive Legal Training System...")
        
        if not DEPENDENCIES_AVAILABLE:
            logger.error("❌ Required dependencies not available")
            return False
            
        try:
            # Initialize embedding model
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.chroma_dir.mkdir(exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("✅ ChromaDB initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    def create_collections(self):
        """Create ChromaDB collections for different data types"""
        logger.info("Creating ChromaDB collections...")
        
        collection_configs = {
            'indian_acts': "Indian Legal Acts and Sections",
            'legal_terms': "Legal Terms and Definitions", 
            'procedures': "Legal Procedures and Processes",
            'forms_templates': "Legal Forms and Templates",
            'jurisdictional': "Jurisdictional and Geographic Information",
            'case_law': "Case Law and Precedents",
            'comprehensive': "All Legal Data Combined"
        }
        
        try:
            for name, description in collection_configs.items():
                # Delete existing collection if it exists
                try:
                    self.chroma_client.delete_collection(name)
                    logger.info(f"Deleted existing collection: {name}")
                except:
                    pass
                
                # Create new collection
                collection = self.chroma_client.create_collection(
                    name=name,
                    metadata={"description": description}
                )
                self.collections[name] = collection
                logger.info(f"✅ Created collection: {name}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collections: {e}")
            return False
    
    def train_indian_acts(self) -> bool:
        """Train on Indian legal acts data"""
        logger.info("🏛️ Training on Indian Legal Acts...")
        
        acts_dir = self.data_sources['indian_acts']
        if not acts_dir.exists():
            logger.warning(f"⚠️ Indian acts directory not found: {acts_dir}")
            return False
        
        collection = self.collections['indian_acts']
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        act_folders = [d for d in acts_dir.iterdir() if d.is_dir()]
        
        for act_folder in tqdm(act_folders, desc="Processing Acts"):
            act_name = act_folder.name.replace('_', ' ').title()
            logger.info(f"Processing {act_name}...")
            
            # Look for structured JSON files
            json_files = list(act_folder.glob("*.json"))
            
            for json_file in json_files:
                if json_file.name.endswith(('_structured.json', '_sections.json', '_final.json')):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Process structured data
                        if isinstance(data, dict) and 'sections' in data:
                            sections = data['sections']
                            if isinstance(sections, list):
                                for i, section in enumerate(sections):
                                    if isinstance(section, dict):
                                        text = section.get('full_text', section.get('content', ''))
                                        if text and len(text.strip()) > 10:
                                            all_documents.append(text)
                                            all_metadatas.append({
                                                'act_name': act_name,
                                                'source_file': json_file.name,
                                                'section_number': str(i + 1),
                                                'data_type': 'indian_act_section',
                                                'extraction_date': data.get('metadata', {}).get('extraction_date', 'unknown')
                                            })
                                            all_ids.append(f"{act_name}_{json_file.stem}_{i}")
                            
                            elif isinstance(sections, dict):
                                for section_num, section_data in sections.items():
                                    if isinstance(section_data, dict):
                                        text = section_data.get('content', section_data.get('full_text', ''))
                                        if text and len(text.strip()) > 10:
                                            all_documents.append(text)
                                            all_metadatas.append({
                                                'act_name': act_name,
                                                'source_file': json_file.name,
                                                'section_number': str(section_num),
                                                'data_type': 'indian_act_section',
                                                'extraction_date': data.get('metadata', {}).get('extraction_date', 'unknown')
                                            })
                                            all_ids.append(f"{act_name}_{json_file.stem}_{section_num}")
                        
                        # Process simple list format
                        elif isinstance(data, list):
                            for i, item in enumerate(data):
                                if isinstance(item, dict):
                                    text = item.get('text', item.get('content', item.get('full_text', '')))
                                    if text and len(text.strip()) > 10:
                                        all_documents.append(text)
                                        all_metadatas.append({
                                            'act_name': act_name,
                                            'source_file': json_file.name,
                                            'section_number': str(i + 1),
                                            'data_type': 'indian_act_section'
                                        })
                                        all_ids.append(f"{act_name}_{json_file.stem}_{i}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {json_file}: {e}")
        
        if all_documents:
            logger.info(f"Adding {len(all_documents)} Indian act sections to database...")
            
            # Process in batches
            batch_size = 100
            for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding to ChromaDB"):
                end_idx = min(i + batch_size, len(all_documents))
                batch_docs = all_documents[i:end_idx]
                batch_metadata = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            logger.info(f"✅ Added {len(all_documents)} Indian act sections")
            return True
        else:
            logger.warning("⚠️ No Indian act data found to train")
            return False
    
    def train_legal_terms(self) -> bool:
        """Train on legal terms and definitions"""
        logger.info("📚 Training on Legal Terms...")
        
        terms_dir = self.data_sources['legal_terms']
        if not terms_dir.exists():
            logger.warning(f"⚠️ Legal terms directory not found: {terms_dir}")
            return False
        
        collection = self.collections['legal_terms']
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Process different legal terms files
        term_files = [
            'legal_terms_structured.json',
            'legal_terms_dictionary.json',
            'legal_terms_by_category.json',
            'legal_terms_simple.json'
        ]
        
        for term_file in term_files:
            file_path = terms_dir / term_file
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"Processing {term_file}...")
                
                if term_file == 'legal_terms_dictionary.json':
                    # Simple dictionary format
                    for term, definition in data.items():
                        if definition and len(definition.strip()) > 5:
                            document = f"Legal Term: {term}\nDefinition: {definition}"
                            all_documents.append(document)
                            all_metadatas.append({
                                'term': term,
                                'data_type': 'legal_term',
                                'source_file': term_file,
                                'category': 'general'
                            })
                            all_ids.append(f"term_{term}_{term_file}")
                
                elif term_file == 'legal_terms_by_category.json':
                    # Categorized format
                    for category, terms in data.items():
                        if isinstance(terms, list):
                            for i, term_data in enumerate(terms):
                                if isinstance(term_data, dict):
                                    definition = term_data.get('definition', '')
                                    term_name = term_data.get('term', f"Term_{i}")
                                    if definition and len(definition.strip()) > 5:
                                        document = f"Legal Term: {term_name}\nCategory: {category}\nDefinition: {definition}"
                                        all_documents.append(document)
                                        all_metadatas.append({
                                            'term': term_name,
                                            'data_type': 'legal_term',
                                            'source_file': term_file,
                                            'category': category
                                        })
                                        all_ids.append(f"term_{category}_{i}_{term_file}")
                
                elif term_file == 'legal_terms_structured.json':
                    # Structured format with metadata
                    if isinstance(data, dict) and 'terms' in data:
                        terms_list = data['terms']
                        for i, term_data in enumerate(terms_list):
                            if isinstance(term_data, dict):
                                term_name = term_data.get('term', f"Term_{i}")
                                definition = term_data.get('definition', '')
                                category = term_data.get('category', 'general')
                                
                                if definition and len(definition.strip()) > 5:
                                    document = f"Legal Term: {term_name}\nCategory: {category}\nDefinition: {definition}"
                                    all_documents.append(document)
                                    all_metadatas.append({
                                        'term': term_name,
                                        'data_type': 'legal_term',
                                        'source_file': term_file,
                                        'category': category
                                    })
                                    all_ids.append(f"term_structured_{i}")
                
                elif term_file == 'legal_terms_simple.json':
                    # Simple list format
                    if isinstance(data, list):
                        for i, term_data in enumerate(data):
                            if isinstance(term_data, dict):
                                term_name = term_data.get('term', f"Term_{i}")
                                definition = term_data.get('definition', '')
                                
                                if definition and len(definition.strip()) > 5:
                                    document = f"Legal Term: {term_name}\nDefinition: {definition}"
                                    all_documents.append(document)
                                    all_metadatas.append({
                                        'term': term_name,
                                        'data_type': 'legal_term',
                                        'source_file': term_file,
                                        'category': 'general'
                                    })
                                    all_ids.append(f"term_simple_{i}")
            
            except Exception as e:
                logger.warning(f"⚠️ Error processing {term_file}: {e}")
        
        if all_documents:
            logger.info(f"Adding {len(all_documents)} legal terms to database...")
            
            # Process in batches
            batch_size = 100
            for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding to ChromaDB"):
                end_idx = min(i + batch_size, len(all_documents))
                batch_docs = all_documents[i:end_idx]
                batch_metadata = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            logger.info(f"✅ Added {len(all_documents)} legal terms")
            return True
        else:
            logger.warning("⚠️ No legal terms data found to train")
            return False
    
    def train_procedures(self) -> bool:
        """Train on legal procedures"""
        logger.info("⚖️ Training on Legal Procedures...")
        
        procedures_dir = self.data_sources['procedures']
        if not procedures_dir.exists():
            logger.warning(f"⚠️ Procedures directory not found: {procedures_dir}")
            return False
        
        collection = self.collections['procedures']
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Process JSON procedure files
        json_files = list(procedures_dir.glob("*.json"))
        
        for json_file in tqdm(json_files, desc="Processing Procedures"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                procedure_name = json_file.stem.replace('_', ' ').title()
                logger.info(f"Processing {procedure_name}...")
                
                # Create comprehensive procedure document
                procedure_text = f"Legal Procedure: {procedure_name}\n\n"
                
                if isinstance(data, dict):
                    if 'procedure_type' in data:
                        procedure_text += f"Type: {data['procedure_type']}\n\n"
                    
                    if 'legal_framework' in data:
                        procedure_text += "Legal Framework:\n"
                        framework = data['legal_framework']
                        for key, value in framework.items():
                            if isinstance(value, list):
                                procedure_text += f"- {key.replace('_', ' ').title()}: {', '.join(value)}\n"
                            else:
                                procedure_text += f"- {key.replace('_', ' ').title()}: {value}\n"
                        procedure_text += "\n"
                    
                    if 'applicable_scenarios' in data:
                        procedure_text += "Applicable Scenarios:\n"
                        for scenario in data['applicable_scenarios']:
                            procedure_text += f"- {scenario}\n"
                        procedure_text += "\n"
                    
                    if 'steps' in data:
                        procedure_text += "Procedure Steps:\n"
                        for i, step in enumerate(data['steps'], 1):
                            procedure_text += f"{i}. {step}\n"
                        procedure_text += "\n"
                    
                    if 'required_documents' in data:
                        procedure_text += "Required Documents:\n"
                        for doc in data['required_documents']:
                            procedure_text += f"- {doc}\n"
                        procedure_text += "\n"
                    
                    if 'timeline' in data:
                        procedure_text += "Timeline:\n"
                        timeline = data['timeline']
                        for key, value in timeline.items():
                            procedure_text += f"- {key.replace('_', ' ').title()}: {value}\n"
                        procedure_text += "\n"
                    
                    if 'fees' in data:
                        procedure_text += "Fees:\n"
                        fees = data['fees']
                        for key, value in fees.items():
                            procedure_text += f"- {key.replace('_', ' ').title()}: {value}\n"
                        procedure_text += "\n"
                
                all_documents.append(procedure_text)
                all_metadatas.append({
                    'procedure_name': procedure_name,
                    'procedure_type': data.get('procedure_type', 'unknown'),
                    'data_type': 'legal_procedure',
                    'source_file': json_file.name
                })
                all_ids.append(f"procedure_{json_file.stem}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing {json_file}: {e}")
        
        # Also process consumer complaint subdirectory
        consumer_dir = procedures_dir / "consumer_complaint"
        if consumer_dir.exists():
            for file_path in consumer_dir.iterdir():
                if file_path.suffix in ['.txt', '.html']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if len(content.strip()) > 50:
                            document = f"Consumer Complaint Information from {file_path.name}:\n\n{content}"
                            all_documents.append(document)
                            all_metadatas.append({
                                'procedure_name': 'Consumer Complaint',
                                'procedure_type': 'Consumer Protection',
                                'data_type': 'legal_procedure_info',
                                'source_file': file_path.name
                            })
                            all_ids.append(f"consumer_complaint_{file_path.stem}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {file_path}: {e}")
        
        if all_documents:
            logger.info(f"Adding {len(all_documents)} legal procedures to database...")
            
            # Process in batches
            batch_size = 50
            for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding to ChromaDB"):
                end_idx = min(i + batch_size, len(all_documents))
                batch_docs = all_documents[i:end_idx]
                batch_metadata = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            logger.info(f"✅ Added {len(all_documents)} legal procedures")
            return True
        else:
            logger.warning("⚠️ No legal procedures data found to train")
            return False
    
    def train_forms_and_templates(self) -> bool:
        """Train on legal forms and templates"""
        logger.info("📋 Training on Legal Forms and Templates...")
        
        forms_dir = self.data_sources['forms_templates']
        if not forms_dir.exists():
            logger.warning(f"⚠️ Forms directory not found: {forms_dir}")
            return False
        
        collection = self.collections['forms_templates']
        
        json_files = list(forms_dir.glob("*.json"))
        if not json_files:
            logger.warning("⚠️ No forms data found")
            return False
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    for form_name, form_data in data.items():
                        if isinstance(form_data, dict):
                            # Create form description
                            form_text = f"Legal Form: {form_name}\n\n"
                            
                            # Process form structure
                            for section, content in form_data.items():
                                form_text += f"{section.replace('_', ' ').title()}:\n"
                                if isinstance(content, dict):
                                    for key, value in content.items():
                                        form_text += f"- {key}: {value}\n"
                                elif isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict):
                                            for k, v in item.items():
                                                form_text += f"- {k}: {v}\n"
                                        else:
                                            form_text += f"- {item}\n"
                                else:
                                    form_text += f"{content}\n"
                                form_text += "\n"
                            
                            all_documents.append(form_text)
                            all_metadatas.append({
                                'form_name': form_name,
                                'data_type': 'legal_form',
                                'source_file': json_file.name
                            })
                            all_ids.append(f"form_{form_name}_{json_file.stem}")
                            
            except Exception as e:
                logger.warning(f"⚠️ Error processing {json_file}: {e}")
        
        if all_documents:
            logger.info(f"Adding {len(all_documents)} legal forms to database...")
            collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"✅ Added {len(all_documents)} legal forms")
            return True
        else:
            logger.warning("⚠️ No forms data found to train")
            return False
    
    def train_jurisdictional_info(self) -> bool:
        """Train on jurisdictional information"""
        logger.info("🗺️ Training on Jurisdictional Information...")
        
        jurisdictional_dir = self.data_sources['jurisdictional']
        if not jurisdictional_dir.exists():
            logger.warning(f"⚠️ Jurisdictional directory not found: {jurisdictional_dir}")
            return False
        
        collection = self.collections['jurisdictional']
        
        json_files = list(jurisdictional_dir.glob("*.json"))
        if not json_files:
            logger.warning("⚠️ No jurisdictional data found")
            return False
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process jurisdictional data
                def process_jurisdictional_section(section_name, section_data, parent_path=""):
                    path = f"{parent_path}/{section_name}" if parent_path else section_name
                    
                    if isinstance(section_data, dict):
                        if 'overview' in section_data:
                            # This is a content section
                            document = f"Jurisdictional Information: {section_name}\n\n"
                            document += f"Overview: {section_data['overview']}\n\n"
                            
                            if 'examples' in section_data:
                                document += "Examples:\n"
                                examples = section_data['examples']
                                if isinstance(examples, dict):
                                    for key, value in examples.items():
                                        document += f"- {key}: {value}\n"
                                document += "\n"
                            
                            all_documents.append(document)
                            all_metadatas.append({
                                'section': section_name,
                                'data_type': 'jurisdictional_info',
                                'source_file': json_file.name,
                                'path': path
                            })
                            all_ids.append(f"jurisdictional_{path.replace('/', '_')}_{json_file.stem}")
                        
                        else:
                            # Recurse through nested structure
                            for key, value in section_data.items():
                                process_jurisdictional_section(key, value, path)
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        process_jurisdictional_section(key, value)
                        
            except Exception as e:
                logger.warning(f"⚠️ Error processing {json_file}: {e}")
        
        if all_documents:
            logger.info(f"Adding {len(all_documents)} jurisdictional documents to database...")
            collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"✅ Added {len(all_documents)} jurisdictional documents")
            return True
        else:
            logger.warning("⚠️ No jurisdictional data found to train")
            return False
    
    def create_comprehensive_collection(self):
        """Create a comprehensive collection with all data combined"""
        logger.info("🔄 Creating comprehensive collection with all data...")
        
        comprehensive_collection = self.collections['comprehensive']
        
        # Combine data from all other collections
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for collection_name, collection in self.collections.items():
            if collection_name == 'comprehensive':
                continue
            
            try:
                # Get all documents from this collection
                results = collection.get()
                
                if results['documents']:
                    for i, (doc, metadata, doc_id) in enumerate(zip(
                        results['documents'], 
                        results['metadatas'], 
                        results['ids']
                    )):
                        all_documents.append(doc)
                        # Add collection source to metadata
                        metadata['source_collection'] = collection_name
                        all_metadatas.append(metadata)
                        all_ids.append(f"comprehensive_{collection_name}_{doc_id}")
                
                logger.info(f"✅ Copied {len(results['documents'])} documents from {collection_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error copying from {collection_name}: {e}")
        
        if all_documents:
            logger.info(f"Adding {len(all_documents)} documents to comprehensive collection...")
            
            # Process in batches
            batch_size = 100
            for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding to Comprehensive Collection"):
                end_idx = min(i + batch_size, len(all_documents))
                batch_docs = all_documents[i:end_idx]
                batch_metadata = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                comprehensive_collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
            
            logger.info(f"✅ Created comprehensive collection with {len(all_documents)} documents")
            return True
        else:
            logger.warning("⚠️ No data found for comprehensive collection")
            return False
    
    def generate_training_report(self):
        """Generate a training report"""
        logger.info("📊 Generating training report...")
        
        report = {
            'training_date': datetime.now().isoformat(),
            'collections': {},
            'total_documents': 0,
            'data_sources_processed': []
        }
        
        for collection_name, collection in self.collections.items():
            try:
                count = collection.count()
                report['collections'][collection_name] = count
                report['total_documents'] += count
                logger.info(f"Collection '{collection_name}': {count} documents")
            except Exception as e:
                logger.warning(f"⚠️ Error getting count for {collection_name}: {e}")
                report['collections'][collection_name] = 0
        
        # Check which data sources were processed
        for source_name, source_path in self.data_sources.items():
            if source_path.exists():
                report['data_sources_processed'].append(source_name)
        
        # Save report
        report_file = Path("training_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Training report saved to {report_file}")
        logger.info(f"📊 Total documents trained: {report['total_documents']}")
        
        return report
    
    def run_full_training(self, rebuild_all=False, acts_only=False, terms_only=False):
        """Run the complete training process"""
        logger.info("🚀 Starting Comprehensive Legal System Training...")
        
        # Initialize system
        if not self.initialize_system():
            return False
        
        # Create collections
        if not self.create_collections():
            return False
        
        success_count = 0
        
        # Train different data types based on flags
        if acts_only:
            logger.info("Training only Indian Acts data...")
            if self.train_indian_acts():
                success_count += 1
        elif terms_only:
            logger.info("Training only Legal Terms data...")
            if self.train_legal_terms():
                success_count += 1
        else:
            # Train all data types
            training_tasks = [
                ("Indian Acts", self.train_indian_acts),
                ("Legal Terms", self.train_legal_terms),
                ("Legal Procedures", self.train_procedures),
                ("Forms and Templates", self.train_forms_and_templates),
                ("Jurisdictional Info", self.train_jurisdictional_info)
            ]
            
            for task_name, task_func in training_tasks:
                logger.info(f"Starting {task_name} training...")
                if task_func():
                    success_count += 1
                    logger.info(f"✅ {task_name} training completed")
                else:
                    logger.warning(f"⚠️ {task_name} training had issues")
        
        # Create comprehensive collection if not training specific types only
        if not acts_only and not terms_only:
            if self.create_comprehensive_collection():
                success_count += 1
        
        # Generate report
        report = self.generate_training_report()
        
        if success_count > 0:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"✅ {success_count} training tasks completed")
            return True
        else:
            logger.error("❌ Training failed - no data was successfully processed")
            return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Train the Comprehensive Legal System')
    parser.add_argument('--rebuild-all', action='store_true', 
                       help='Rebuild all collections from scratch')
    parser.add_argument('--acts-only', action='store_true',
                       help='Train only Indian Acts data')
    parser.add_argument('--terms-only', action='store_true',
                       help='Train only Legal Terms data')
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("❌ Required dependencies not available")
        logger.error("Please install: pip install chromadb sentence-transformers pandas tqdm")
        return False
    
    # Print header
    print("=" * 70)
    print("🏛️  AI Legal Assistant - Comprehensive Training System")
    print("=" * 70)
    print("Training comprehensive legal knowledge base...")
    print()
    
    # Initialize trainer
    trainer = ComprehensiveLegalTrainer()
    
    # Run training
    success = trainer.run_full_training(
        rebuild_all=args.rebuild_all,
        acts_only=args.acts_only,
        terms_only=args.terms_only
    )
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("The AI Legal Assistant has been trained with comprehensive legal data.")
        print("You can now run the system with:")
        print("  python enhanced_streamlit_app.py")
        print("=" * 70)
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED")
        print("=" * 70)
        print("Please check the logs and ensure all data files are present.")
        print("=" * 70)
        return False

if __name__ == "__main__":
    main()
