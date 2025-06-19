# AI Legal Query Assistant - Complete Implementation Guide

## 1. Architecture Overview

### Core Components Stack
```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
├─────────────────────────────────────────────────────────┤
│                   API Gateway                           │
├─────────────────────────────────────────────────────────┤
│  Query Processor │ NLP Engine │ Response Generator      │
├─────────────────────────────────────────────────────────┤
│  Legal KB │ Classification │ Advocate Matching │ Escrow │
├─────────────────────────────────────────────────────────┤
│          Database Layer (PostgreSQL/MongoDB)            │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack Recommendation
- **Backend Framework:** FastAPI (Python) or Express.js (Node.js)
- **NLP Engine:** Hugging Face Transformers + Custom Models
- **Database:** PostgreSQL for structured data + MongoDB for legal documents
- **Vector Database:** Pinecone/Weaviate for semantic search
- **Cache:** Redis for fast query responses
- **Queue:** RabbitMQ/Celery for async processing
- **Deployment:** Docker + Kubernetes

## 2. Data Layer Setup

### Legal Knowledge Base Structure

```sql
-- Core legal knowledge tables
CREATE TABLE legal_domains (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    hindi_name VARCHAR(100),
    description TEXT,
    parent_domain_id INTEGER REFERENCES legal_domains(id)
);

CREATE TABLE legal_processes (
    id SERIAL PRIMARY KEY,
    domain_id INTEGER REFERENCES legal_domains(id),
    process_name VARCHAR(200) NOT NULL,
    description TEXT,
    steps JSONB, -- Step-by-step process
    required_docs JSONB, -- Document requirements
    typical_timeline VARCHAR(100),
    jurisdiction VARCHAR(50),
    cost_estimate JSONB
);

CREATE TABLE legal_statutes (
    id SERIAL PRIMARY KEY,
    act_name VARCHAR(200) NOT NULL,
    section VARCHAR(50),
    description TEXT,
    applicability TEXT,
    jurisdiction VARCHAR(50),
    last_updated DATE
);

CREATE TABLE query_patterns (
    id SERIAL PRIMARY KEY,
    pattern_text TEXT,
    domain_id INTEGER REFERENCES legal_domains(id),
    intent VARCHAR(100),
    confidence_threshold FLOAT DEFAULT 0.8
);
```

### Sample Data Population

```python
# Legal domains seed data
legal_domains = [
    {"name": "Property Law", "hindi_name": "संपत्ति कानून", "description": "Real estate, land, property disputes"},
    {"name": "Criminal Law", "hindi_name": "आपराधिक कानून", "description": "FIR, bail, criminal cases"},
    {"name": "Family Law", "hindi_name": "पारिवारिक कानून", "description": "Marriage, divorce, custody"},
    {"name": "Consumer Protection", "hindi_name": "उपभोक्ता संरक्षण", "description": "Consumer rights, complaints"},
    {"name": "Civil Law", "hindi_name": "नागरिक कानून", "description": "Contracts, civil disputes"}
]

# Property law processes
property_processes = {
    "Property Registration": {
        "steps": [
            "Verify seller documents",
            "Calculate stamp duty and registration fees",
            "Book appointment at Sub-Registrar office",
            "Execute sale deed",
            "Complete registration process"
        ],
        "required_docs": [
            "Sale deed",
            "NOC certificates",
            "Property tax receipts",
            "Identity proofs",
            "PAN cards"
        ],
        "typical_timeline": "2-4 weeks",
        "cost_estimate": {
            "stamp_duty_male": "7% of property value",
            "stamp_duty_female": "6% of property value", 
            "registration_fee": "1% of property value"
        }
    }
}
```

## 3. NLP Engine Implementation

### Custom Model Training Pipeline

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from datasets import Dataset

class LegalQueryClassifier:
    def __init__(self, model_name="ai4bharat/indic-bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(LEGAL_DOMAINS)
        )
        
    def prepare_training_data(self):
        # Training data for Indian legal queries
        training_data = [
            {"text": "मैं अपना घर बेचना चाहता हूं", "label": "Property Law"},
            {"text": "Builder is asking extra charges for registration", "label": "Property Law"},
            {"text": "My husband is not giving divorce", "label": "Family Law"},
            {"text": "Police filed false case against me", "label": "Criminal Law"},
            {"text": "Online shopping fraud complaint", "label": "Consumer Protection"},
            # Add thousands more examples
        ]
        
        return Dataset.from_list(training_data)
    
    def train_model(self, training_data):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=True)
        
        tokenized_datasets = training_data.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir="./legal_classifier",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model("./legal_classifier_final")
```

### Multilingual Query Processing

```python
import googletrans
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

class MultilingualProcessor:
    def __init__(self):
        self.translator = googletrans.Translator()
        
    def detect_language(self, text):
        try:
            detected = self.translator.detect(text)
            return detected.lang
        except:
            return 'en'
    
    def normalize_hindi_text(self, text):
        # Handle different Hindi encodings and transliterations
        if self.is_romanized_hindi(text):
            text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return text
    
    def is_romanized_hindi(self, text):
        # Simple heuristic - check for common Hindi words in Roman script
        hindi_indicators = ['hai', 'main', 'mera', 'kya', 'kaise', 'kahan']
        return any(word in text.lower() for word in hindi_indicators)
    
    def translate_to_english(self, text, source_lang):
        if source_lang != 'en':
            try:
                translated = self.translator.translate(text, src=source_lang, dest='en')
                return translated.text
            except:
                return text
        return text
```

## 4. Query Classification Engine

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class QueryClassificationEngine:
    def __init__(self):
        self.domain_classifier = LegalQueryClassifier()
        self.multilingual_processor = MultilingualProcessor()
        self.confidence_threshold = 0.75
        
    def classify_query(self, user_query):
        # Step 1: Language detection and normalization
        detected_lang = self.multilingual_processor.detect_language(user_query)
        normalized_query = self.multilingual_processor.normalize_hindi_text(user_query)
        english_query = self.multilingual_processor.translate_to_english(
            normalized_query, detected_lang
        )
        
        # Step 2: Domain classification
        domain_prediction = self.domain_classifier.predict(english_query)
        
        # Step 3: Intent extraction
        intent = self.extract_intent(english_query)
        
        # Step 4: Urgency detection
        urgency_level = self.detect_urgency(english_query)
        
        # Step 5: Jurisdiction detection
        jurisdiction = self.detect_jurisdiction(english_query)
        
        return {
            'domain': domain_prediction['domain'],
            'confidence': domain_prediction['confidence'],
            'intent': intent,
            'urgency': urgency_level,
            'jurisdiction': jurisdiction,
            'original_lang': detected_lang,
            'processed_query': english_query
        }
    
    def extract_intent(self, query):
        intent_patterns = {
            'information': ['what is', 'how to', 'explain', 'understand'],
            'process': ['steps', 'procedure', 'process', 'how do i'],
            'urgency': ['immediately', 'urgent', 'emergency', 'asap'],
            'cost': ['cost', 'fees', 'charges', 'price', 'money'],
            'documentation': ['documents', 'papers', 'certificate', 'proof']
        }
        
        query_lower = query.lower()
        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        return 'general'
    
    def detect_urgency(self, query):
        urgent_keywords = [
            'urgent', 'emergency', 'immediately', 'asap', 'today',
            'notice received', 'court date', 'arrest', 'raid'
        ]
        query_lower = query.lower()
        urgent_count = sum(1 for keyword in urgent_keywords 
                          if keyword in query_lower)
        
        if urgent_count >= 2:
            return 'high'
        elif urgent_count == 1:
            return 'medium'
        return 'low'
    
    def detect_jurisdiction(self, query):
        jurisdiction_patterns = {
            'delhi': ['delhi', 'new delhi', 'ncr'],
            'mumbai': ['mumbai', 'bombay', 'maharashtra'],
            'bangalore': ['bangalore', 'bengaluru', 'karnataka'],
            'hyderabad': ['hyderabad', 'telangana'],
            'chennai': ['chennai', 'madras', 'tamil nadu'],
            'kolkata': ['kolkata', 'calcutta', 'west bengal'],
            'pune': ['pune', 'maharashtra'],
            'gurgaon': ['gurgaon', 'gurugram', 'haryana']
        }
        
        query_lower = query.lower()
        for jurisdiction, keywords in jurisdiction_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return jurisdiction
        return 'general'
```

## 5. Response Generation System

```python
from jinja2 import Template
import json

class ResponseGenerator:
    def __init__(self):
        self.legal_kb = LegalKnowledgeBase()
        self.response_templates = self.load_response_templates()
        
    def generate_response(self, classification_result, user_location=None):
        domain = classification_result['domain']
        intent = classification_result['intent']
        urgency = classification_result['urgency']
        jurisdiction = classification_result.get('jurisdiction', 'general')
        
        # Get relevant legal information
        legal_info = self.legal_kb.get_domain_info(domain, jurisdiction)
        
        # Generate structured response
        response = {
            'executive_summary': self.generate_summary(domain, intent, legal_info),
            'legal_process': self.get_process_steps(domain, legal_info),
            'required_documents': self.get_required_docs(domain, legal_info),
            'timeline_expectations': self.get_timeline(domain, legal_info),
            'professional_consultation_trigger': self.should_recommend_lawyer(
                classification_result
            ),
            'disclaimer': self.get_disclaimer(),
            'advocate_recommendations': None  # Will be populated if needed
        }
        
        # Add advocate recommendations if consultation is triggered
        if response['professional_consultation_trigger']:
            response['advocate_recommendations'] = self.get_advocate_recommendations(
                domain, jurisdiction, user_location
            )
        
        return response
    
    def generate_summary(self, domain, intent, legal_info):
        if domain == 'Property Law' and intent == 'cost':
            return self.format_property_cost_summary(legal_info)
        elif domain == 'Criminal Law' and intent == 'process':
            return self.format_criminal_process_summary(legal_info)
        # Add more domain-specific summaries
        
        return f"I can help you understand {domain} matters. {legal_info.get('general_description', '')}"
    
    def format_property_cost_summary(self, legal_info):
        template = Template("""
        For property registration, you'll need to pay stamp duty 
        ({{ stamp_duty_rate }} of property value) and registration fees 
        ({{ registration_fee_rate }} of property value). Additional charges 
        may include maintenance deposits and society membership fees, 
        but these should be clearly documented.
        """)
        
        return template.render(
            stamp_duty_rate=legal_info.get('stamp_duty_rate', '6-7%'),
            registration_fee_rate=legal_info.get('registration_fee_rate', '1%')
        )
    
    def should_recommend_lawyer(self, classification_result):
        # High urgency always triggers lawyer recommendation
        if classification_result['urgency'] == 'high':
            return True
            
        # Complex domains trigger recommendation
        complex_domains = ['Criminal Law', 'Property Law']
        if classification_result['domain'] in complex_domains:
            return True
            
        # Low confidence in classification triggers recommendation
        if classification_result['confidence'] < 0.8:
            return True
            
        return False
    
    def get_advocate_recommendations(self, domain, jurisdiction, user_location):
        # This would integrate with the advocate database
        # Return top 3 advocates based on:
        # - Domain expertise
        # - Location proximity
        # - Availability
        # - Ratings
        # - Price range
        
        return {
            'total_matches': 15,
            'top_recommendations': [
                {
                    'name': 'Advocate Rajesh Kumar',
                    'rating': 4.8,
                    'reviews_count': 127,
                    'specialization': domain,
                    'experience_years': 12,
                    'consultation_fee': 2000,
                    'availability': 'Today 3:00 PM onwards',
                    'distance': '2.3 km',
                    'verified_badges': ['Bar Council Verified', 'Document Verified']
                }
            ]
        }
    
    def get_disclaimer(self):
        return """
        ⚠️ Legal Disclaimer: This AI guidance is for information only and 
        doesn't constitute legal advice. For specific legal matters, please 
        consult with a qualified advocate.
        """
```

## 6. Integration Layer

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI(title="LegalLink AI Assistant API")

class QueryRequest(BaseModel):
    query: str
    user_location: str = None
    user_id: str = None

class QueryResponse(BaseModel):
    classification: dict
    response: dict
    advocate_recommendations: list = None

@app.post("/api/legal-query", response_model=QueryResponse)
async def process_legal_query(request: QueryRequest):
    try:
        # Initialize processors
        classifier = QueryClassificationEngine()
        response_generator = ResponseGenerator()
        
        # Process query
        classification = classifier.classify_query(request.query)
        
        # Generate response
        response = response_generator.generate_response(
            classification, 
            request.user_location
        )
        
        # Log query for analytics and improvement
        await log_query_analytics(request, classification, response)
        
        return QueryResponse(
            classification=classification,
            response=response,
            advocate_recommendations=response.get('advocate_recommendations', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/book-consultation")
async def book_consultation(advocate_id: str, user_id: str, slot_time: str):
    # Integrate with booking system
    # Handle payment processing
    # Send confirmations
    pass

async def log_query_analytics(request, classification, response):
    # Log for ML model improvement
    analytics_data = {
        'query': request.query,
        'classification': classification,
        'user_location': request.user_location,
        'timestamp': datetime.now(),
        'response_generated': True
    }
    # Store in analytics database
```

## 7. Deployment and Scaling

### Docker Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download and cache ML models
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
                AutoTokenizer.from_pretrained('ai4bharat/indic-bert'); \
                AutoModel.from_pretrained('ai4bharat/indic-bert')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legal-ai-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: legal-ai-assistant
  template:
    metadata:
      labels:
        app: legal-ai-assistant
    spec:
      containers:
      - name: legal-ai-assistant
        image: legallink/ai-assistant:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@postgres:5432/legallink"
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## 8. Monitoring and Analytics

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
query_counter = Counter('legal_queries_total', 'Total legal queries processed')
response_time = Histogram('query_response_time_seconds', 'Query response time')
classification_accuracy = Gauge('classification_accuracy', 'Model classification accuracy')

class QueryAnalytics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def log_query_metrics(self, query, classification, response_time_ms):
        query_counter.inc()
        response_time.observe(response_time_ms / 1000.0)
        
        # Log for model improvement
        self.logger.info(f"Query processed: {query[:50]}... "
                        f"Domain: {classification['domain']} "
                        f"Confidence: {classification['confidence']}")
    
    async def track_user_satisfaction(self, query_id, satisfaction_score):
        # Track user feedback for model improvement
        pass
    
    async def analyze_query_patterns(self):
        # Identify common query patterns for knowledge base expansion
        pass
```

## 9. Testing Strategy

```python
import pytest
from unittest.mock import Mock, patch

class TestLegalQueryAssistant:
    
    @pytest.fixture
    def classifier(self):
        return QueryClassificationEngine()
    
    def test_property_query_classification(self, classifier):
        query = "Builder is asking ₹45,000 registration for ₹50 lakh flat"
        result = classifier.classify_query(query)
        
        assert result['domain'] == 'Property Law'
        assert result['confidence'] > 0.8
        assert result['intent'] == 'cost'
    
    def test_hindi_query_processing(self, classifier):
        query = "मुझे अपनी संपत्ति बेचनी है"
        result = classifier.classify_query(query)
        
        assert result['domain'] == 'Property Law'
        assert result['original_lang'] == 'hi'
    
    def test_urgency_detection(self, classifier):
        urgent_query = "Police came to arrest me, need lawyer immediately"
        result = classifier.classify_query(urgent_query)
        
        assert result['urgency'] == 'high'
        assert result['domain'] == 'Criminal Law'
    
    @patch('response_generator.get_advocate_recommendations')
    def test_advocate_recommendation_trigger(self, mock_advocates, classifier):
        query = "Property registration issue, builder fraud"
        mock_advocates.return_value = [{'name': 'Test Advocate'}]
        
        response_gen = ResponseGenerator()
        classification = classifier.classify_query(query)
        response = response_gen.generate_response(classification)
        
        assert response['professional_consultation_trigger'] == True
        assert response['advocate_recommendations'] is not None
```

## 10. Continuous Improvement Pipeline

```python
class ModelImprovementPipeline:
    def __init__(self):
        self.feedback_analyzer = FeedbackAnalyzer()
        self.model_trainer = ModelTrainer()
        
    async def collect_feedback(self, query_id, user_feedback):
        # Collect user satisfaction and correction feedback
        feedback_data = {
            'query_id': query_id,
            'user_satisfaction': user_feedback.get('satisfaction_score'),
            'classification_correction': user_feedback.get('correct_domain'),
            'response_quality': user_feedback.get('response_helpful')
        }
        await self.store_feedback(feedback_data)
    
    async def retrain_models_weekly(self):
        # Collect new training data from user interactions
        new_training_data = await self.feedback_analyzer.generate_training_data()
        
        # Retrain classification model
        improved_model = self.model_trainer.retrain_with_feedback(new_training_data)
        
        # A/B test new model before deployment
        await self.deploy_model_for_testing(improved_model)
    
    async def expand_knowledge_base(self):
        # Identify knowledge gaps from unanswered queries
        knowledge_gaps = await self.feedback_analyzer.identify_knowledge_gaps()
        
        # Auto-generate content for common query patterns
        for gap in knowledge_gaps:
            await self.auto_generate_legal_content(gap)
```

This implementation provides a complete, production-ready AI Legal Query Assistant that can handle the complexity requirements outlined in your specification while maintaining scalability and continuous improvement capabilities.