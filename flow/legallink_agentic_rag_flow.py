import asyncio
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ollama
import google.generativeai as genai
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

class QueryType(Enum):
    CONSUMER_PROTECTION = "consumer_protection"
    CRIMINAL_LAW = "criminal_law"
    FAMILY_LAW = "family_law"
    PROPERTY_LAW = "property_law"
    CIVIL_LAW = "civil_law"
    URGENT_THREAT = "urgent_threat"

class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QueryClassification:
    query_type: QueryType
    urgency: UrgencyLevel
    confidence: float
    requires_legal_counsel: bool
    jurisdiction: str
    keywords: List[str]

@dataclass
class LegalResponse:
    executive_summary: str
    legal_process: List[str]
    required_documents: List[str]
    timeline: str
    red_flags: List[str]
    advocate_recommendation: bool
    disclaimer: str
    relevant_laws: List[str]

class LegalLinkRAGAssistant:
    def __init__(self):
        # Initialize models and services
        self.ollama_client = ollama.Client()
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Initialize vector database
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("legal_knowledge")
        
        # Sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Legal knowledge base
        self.legal_kb = self._load_legal_knowledge()
        
    def _load_legal_knowledge(self) -> Dict:
        """Load basic legal knowledge into memory"""
        return {
            "consumer_protection": {
                "laws": ["Consumer Protection Act 2019", "Legal Metrology Act 2009"],
                "procedures": ["File complaint with consumer forum", "Gather purchase evidence", "Calculate compensation"],
                "timelines": "30-90 days for district forum",
                "documents": ["Purchase receipt", "Product photos", "Communication records"]
            },
            "criminal_threats": {
                "laws": ["IPC Section 384 (Extortion)", "IPC Section 506 (Criminal Intimidation)", "Protection of Women from Domestic Violence Act"],
                "procedures": ["File police complaint", "Gather evidence", "Seek legal protection"],
                "timelines": "Immediate action required",
                "documents": ["Written threats", "Audio/video evidence", "Witness statements"]
            }
        }

    async def classify_query(self, query: str) -> QueryClassification:
        """Classify the user query using Ollama 3.2B"""
        
        classification_prompt = f"""
        Classify this legal query into appropriate categories:
        Query: "{query}"
        
        Analyze for:
        1. Legal domain (consumer protection, criminal law, family law, property law, civil law)
        2. Urgency level (1-4, where 4 is critical/immediate threat)
        3. Whether immediate legal counsel is required
        4. Key legal concepts mentioned
        
        Respond in JSON format with: query_type, urgency_level, requires_counsel, keywords, jurisdiction
        """
        
        response = self.ollama_client.generate(
            model="llama3.2:3b",
            prompt=classification_prompt
        )
        
        # Parse classification (simplified - add error handling in production)
        try:
            classification_data = json.loads(response['response'])
            
            # Map to our enums
            query_type_map = {
                "consumer protection": QueryType.CONSUMER_PROTECTION,
                "criminal law": QueryType.CRIMINAL_LAW,
                "family law": QueryType.FAMILY_LAW,
                "property law": QueryType.PROPERTY_LAW,
                "civil law": QueryType.CIVIL_LAW
            }
            
            # Detect urgent threats
            threat_keywords = ["rape case", "false case", "extortion", "blackmail", "threat"]
            if any(keyword in query.lower() for keyword in threat_keywords):
                urgency = UrgencyLevel.CRITICAL
                query_type = QueryType.CRIMINAL_LAW
                requires_counsel = True
            else:
                query_type = query_type_map.get(classification_data.get("query_type", "").lower(), QueryType.CIVIL_LAW)
                urgency = UrgencyLevel(min(4, max(1, classification_data.get("urgency_level", 2))))
                requires_counsel = classification_data.get("requires_counsel", False)
            
            return QueryClassification(
                query_type=query_type,
                urgency=urgency,
                confidence=0.85,  # Calculate based on model confidence
                requires_legal_counsel=requires_counsel,
                jurisdiction="pan_india",  # Default, can be refined
                keywords=classification_data.get("keywords", [])
            )
            
        except json.JSONDecodeError:
            # Fallback classification
            return self._fallback_classification(query)

    def _fallback_classification(self, query: str) -> QueryClassification:
        """Fallback classification using keyword matching"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["rape", "false case", "extortion", "blackmail"]):
            return QueryClassification(
                query_type=QueryType.CRIMINAL_LAW,
                urgency=UrgencyLevel.CRITICAL,
                confidence=0.9,
                requires_legal_counsel=True,
                jurisdiction="pan_india",
                keywords=["criminal threat", "extortion"]
            )
        elif any(word in query_lower for word in ["purchase", "bottle", "price", "consumer"]):
            return QueryClassification(
                query_type=QueryType.CONSUMER_PROTECTION,
                urgency=UrgencyLevel.LOW,
                confidence=0.8,
                requires_legal_counsel=False,
                jurisdiction="pan_india",
                keywords=["consumer protection", "overcharging"]
            )
        else:
            return QueryClassification(
                query_type=QueryType.CIVIL_LAW,
                urgency=UrgencyLevel.MEDIUM,
                confidence=0.6,
                requires_legal_counsel=True,
                jurisdiction="pan_india",
                keywords=["general legal"]
            )

    async def search_indian_kanoon(self, keywords: List[str], query_type: QueryType) -> List[Dict]:
        """Search Indian Kanoon for relevant case law"""
        search_results = []
        
        # Construct search query
        search_query = " ".join(keywords)
        if query_type == QueryType.CONSUMER_PROTECTION:
            search_query += " consumer protection act"
        elif query_type == QueryType.CRIMINAL_LAW:
            search_query += " IPC criminal law"
        
        try:
            # Indian Kanoon search URL (simplified)
            search_url = f"https://indiankanoon.org/search/?formInput={search_query.replace(' ', '%20')}"
            
            # Note: In production, implement proper web scraping with rate limiting
            # This is a simplified example
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract case information (adapt based on actual HTML structure)
                cases = soup.find_all('div', class_='result')[:3]  # Top 3 results
                
                for case in cases:
                    title = case.find('a')
                    if title:
                        search_results.append({
                            'title': title.text.strip(),
                            'url': 'https://indiankanoon.org' + title.get('href', ''),
                            'snippet': case.get_text()[:200] + '...'
                        })
            
        except Exception as e:
            print(f"Error searching Indian Kanoon: {e}")
            # Return empty results on error
        
        return search_results

    async def retrieve_relevant_knowledge(self, classification: QueryClassification) -> Dict:
        """Retrieve relevant legal knowledge from local KB and external sources"""
        
        # Get local knowledge
        local_knowledge = {}
        if classification.query_type == QueryType.CONSUMER_PROTECTION:
            local_knowledge = self.legal_kb.get("consumer_protection", {})
        elif classification.query_type == QueryType.CRIMINAL_LAW:
            local_knowledge = self.legal_kb.get("criminal_threats", {})
        
        # Search Indian Kanoon for case law
        case_law = await self.search_indian_kanoon(classification.keywords, classification.query_type)
        
        return {
            "local_knowledge": local_knowledge,
            "case_law": case_law,
            "classification": classification
        }

    async def generate_legal_response(self, query: str, knowledge: Dict) -> LegalResponse:
        """Generate comprehensive legal response using Ollama"""
        
        classification = knowledge["classification"]
        local_kb = knowledge["local_knowledge"]
        case_law = knowledge["case_law"]
        
        # Construct comprehensive prompt
        response_prompt = f"""
        You are a legal information assistant for India. Provide a structured response to this query:
        
        Query: "{query}"
        Query Type: {classification.query_type.value}
        Urgency: {classification.urgency.value}/4
        
        Available Legal Knowledge:
        - Laws: {local_kb.get('laws', [])}
        - Procedures: {local_kb.get('procedures', [])}
        - Required Documents: {local_kb.get('documents', [])}
        - Timeline: {local_kb.get('timelines', 'Variable')}
        
        Relevant Case Law:
        {json.dumps(case_law, indent=2)}
        
        Provide response in this JSON structure:
        {{
            "executive_summary": "Brief 2-3 sentence answer",
            "legal_process": ["Step 1", "Step 2", "Step 3"],
            "required_documents": ["Document 1", "Document 2"],
            "timeline": "Expected timeframe",
            "red_flags": ["Warning 1", "Warning 2"],
            "advocate_recommendation": true/false,
            "relevant_laws": ["Law 1", "Law 2"]
        }}
        
        Important: 
        - If query involves threats/extortion/false cases, mark advocate_recommendation as true
        - Include relevant IPC sections for criminal matters
        - For consumer issues, mention Consumer Protection Act 2019
        """
        
        response = self.ollama_client.generate(
            model="llama3.2:3b",
            prompt=response_prompt
        )
        
        try:
            response_data = json.loads(response['response'])
            return LegalResponse(
                executive_summary=response_data.get("executive_summary", ""),
                legal_process=response_data.get("legal_process", []),
                required_documents=response_data.get("required_documents", []),
                timeline=response_data.get("timeline", ""),
                red_flags=response_data.get("red_flags", []),
                advocate_recommendation=response_data.get("advocate_recommendation", True),
                disclaimer="This is AI-generated legal information, not legal advice. Consult a qualified advocate for your specific situation.",
                relevant_laws=response_data.get("relevant_laws", [])
            )
        except json.JSONDecodeError:
            return self._generate_fallback_response(classification)

    def _generate_fallback_response(self, classification: QueryClassification) -> LegalResponse:
        """Generate fallback response for critical cases"""
        if classification.urgency == UrgencyLevel.CRITICAL:
            return LegalResponse(
                executive_summary="This appears to be a serious legal threat requiring immediate professional help.",
                legal_process=[
                    "Document all threats immediately",
                    "File police complaint under IPC Section 384 (Extortion)",
                    "Consult criminal lawyer immediately",
                    "Gather all evidence"
                ],
                required_documents=["Screenshots/recordings of threats", "Written evidence", "Witness statements"],
                timeline="Immediate action required - within 24 hours",
                red_flags=["Criminal extortion", "False case threats", "Blackmail"],
                advocate_recommendation=True,
                disclaimer="This is AI-generated legal information, not legal advice. Consult a qualified advocate immediately.",
                relevant_laws=["IPC Section 384", "IPC Section 506", "IPC Section 182"]
            )
        else:
            return LegalResponse(
                executive_summary="Please consult with a qualified advocate for proper legal guidance.",
                legal_process=["Gather relevant documents", "Consult with advocate", "Follow legal procedures"],
                required_documents=["Relevant paperwork", "Evidence"],
                timeline="Varies by case",
                red_flags=["Seek professional advice"],
                advocate_recommendation=True,
                disclaimer="This is AI-generated legal information, not legal advice. Consult a qualified advocate for your specific situation.",
                relevant_laws=["Applicable Indian laws"]
            )

    async def simplify_with_gemini(self, response: LegalResponse) -> LegalResponse:
        """Use Gemini to simplify legal language"""
        
        simplification_prompt = f"""
        Simplify this legal response for a common person in India who may not understand legal terminology:
        
        Executive Summary: {response.executive_summary}
        Legal Process: {response.legal_process}
        
        Make it:
        1. Easy to understand (8th grade reading level)
        2. Use simple Hindi-English mixed language where appropriate
        3. Explain legal terms in brackets
        4. Keep the same structure but simpler language
        
        Return in same JSON format with simplified text.
        """
        
        try:
            gemini_response = self.gemini_model.generate_content(simplification_prompt)
            
            # For this example, we'll keep the original response
            # In production, parse Gemini's response and update the LegalResponse
            
            # Simple example of simplification
            simplified_summary = response.executive_summary.replace(
                "pursuant to", "according to"
            ).replace(
                "aforementioned", "mentioned above"
            )
            
            response.executive_summary = simplified_summary
            
        except Exception as e:
            print(f"Gemini simplification failed: {e}")
            # Keep original response if simplification fails
        
        return response

    async def process_query(self, query: str) -> Dict:
        """Main processing pipeline"""
        
        # Step 1: Classify query
        classification = await self.classify_query(query)
        
        # Step 2: Retrieve relevant knowledge
        knowledge = await self.retrieve_relevant_knowledge(classification)
        
        # Step 3: Generate legal response
        legal_response = await self.generate_legal_response(query, knowledge)
        
        # Step 4: Simplify with Gemini
        simplified_response = await self.simplify_with_gemini(legal_response)
        
        # Step 5: Format final response
        return {
            "classification": {
                "query_type": classification.query_type.value,
                "urgency": classification.urgency.value,
                "confidence": classification.confidence,
                "requires_legal_counsel": classification.requires_legal_counsel
            },
            "response": {
                "executive_summary": simplified_response.executive_summary,
                "legal_process": simplified_response.legal_process,
                "required_documents": simplified_response.required_documents,
                "timeline": simplified_response.timeline,
                "red_flags": simplified_response.red_flags,
                "advocate_recommendation": simplified_response.advocate_recommendation,
                "relevant_laws": simplified_response.relevant_laws,
                "disclaimer": simplified_response.disclaimer
            }
        }

# Example usage
async def main():
    assistant = LegalLinkRAGAssistant()
    
    # Test queries
    queries = [
        "I purchased a water bottle for 20 rupees where the actual price of that bottle is 15 rupees. What should I do?",
        "My wife asked me to buy a luxury gift otherwise she will do fake rape case on me. What should I do?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        result = await assistant.process_query(query)
        
        print(f"\nClassification:")
        print(f"- Type: {result['classification']['query_type']}")
        print(f"- Urgency: {result['classification']['urgency']}/4")
        print(f"- Requires Counsel: {result['classification']['requires_legal_counsel']}")
        
        print(f"\nResponse:")
        print(f"Summary: {result['response']['executive_summary']}")
        print(f"Process: {result['response']['legal_process']}")
        print(f"Documents: {result['response']['required_documents']}")
        print(f"Timeline: {result['response']['timeline']}")
        print(f"Red Flags: {result['response']['red_flags']}")
        print(f"Laws: {result['response']['relevant_laws']}")
        print(f"Advocate Needed: {result['response']['advocate_recommendation']}")

if __name__ == "__main__":
    asyncio.run(main())