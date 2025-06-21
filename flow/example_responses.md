# Example Responses from LegalLink RAG Assistant

## Test Case 1: Consumer Protection Issue
**Query:** "I purchased a water bottle for 20 rupees where the actual price of that bottle is 15 rupees. What should I do?"

### System Response:

```json
{
  "classification": {
    "query_type": "consumer_protection",
    "urgency": 1,
    "confidence": 0.92,
    "requires_legal_counsel": false
  },
  "response": {
    "executive_summary": "You have been overcharged by ₹5 for a water bottle. This is a consumer protection issue where you can file a complaint for unfair trade practice and seek refund plus compensation.",
    
    "legal_process": [
      "Keep your purchase receipt and take photos of the bottle with MRP clearly visible",
      "First, approach the shopkeeper politely and ask for refund of extra amount",
      "If refused, file online complaint on National Consumer Helpline (1915) or consumerhelpline.gov.in",
      "For amounts above ₹1 lakh, file complaint in District Consumer Forum",
      "You can claim refund plus compensation for mental harassment and travel costs"
    ],
    
    "required_documents": [
      "Purchase receipt/bill",
      "Photo of product showing MRP",
      "Any communication with seller",
      "Bank statement if paid digitally"
    ],
    
    "timeline": "Online complaint: 7-15 days response, Consumer forum: 30-90 days",
    
    "red_flags": [
      "Shop regularly overcharges customers",
      "No proper billing or receipt given",
      "MRP is covered/modified on products"
    ],
    
    "advocate_recommendation": false,
    
    "relevant_laws": [
      "Consumer Protection Act 2019",
      "Legal Metrology Act 2009",
      "Essential Commodities Act"
    ],
    
    "disclaimer": "This is AI-generated legal information, not legal advice. For complex cases, consult a qualified advocate."
  }
}
```

---

## Test Case 2: Criminal Threat/Extortion
**Query:** "My wife asked me to buy a luxury gift otherwise she will do fake rape case on me. What should I do?"

### System Response:

```json
{
  "classification": {
    "query_type": "criminal_law",
    "urgency": 4,
    "confidence": 0.98,
    "requires_legal_counsel": true
  },
  "response": {
    "executive_summary": "🚨 CRITICAL ALERT: This is criminal extortion and threat of false case filing. You need immediate legal protection and should document everything right now.",
    
    "legal_process": [
      "IMMEDIATELY document this threat - take screenshots, voice recordings if possible",
      "Do NOT comply with the demand - this will only encourage more threats",
      "File police complaint under IPC Section 384 (Extortion) within 24 hours",
      "Contact a criminal lawyer immediately before taking any action",
      "Apply for anticipatory bail as precautionary measure",
      "Gather witnesses who can support your character",
      "Save all communications and evidence securely"
    ],
    
    "required_documents": [
      "Written/audio evidence of threats",
      "WhatsApp messages, emails, call recordings",
      "Witness statements from family/friends",
      "Your financial records showing inability to buy expensive gifts",
      "Marriage certificate and family photos showing normal relationship",
      "Any previous similar threats or patterns"
    ],
    
    "timeline": "IMMEDIATE - File police complaint within 24-48 hours, Anticipatory bail application: 1-7 days",
    
    "red_flags": [
      "🔴 Criminal extortion under IPC Section 384",
      "🔴 Threat of false case filing under IPC Section 182",
      "🔴 Criminal intimidation under IPC Section 506", 
      "🔴 This is a serious criminal offense",
      "🔴 Pattern may escalate if not stopped immediately"
    ],
    
    "advocate_recommendation": true,
    
    "relevant_laws": [
      "IPC Section 384 - Extortion",
      "IPC Section 506 - Criminal Intimidation", 
      "IPC Section 182 - False information to public servant",
      "CrPC Section 438 - Anticipatory Bail",
      "Protection of Women from Domestic Violence Act (if applicable)"
    ],
    
    "disclaimer": "🚨 EMERGENCY LEGAL SITUATION: This is AI-generated guidance for immediate help. CONSULT A CRIMINAL LAWYER IMMEDIATELY. Do not delay - your legal safety is at risk."
  },
  
  "immediate_actions": {
    "advocate_matching_triggered": true,
    "emergency_consultation_available": true,
    "recommended_advocates": [
      {
        "name": "Available within 2 hours",
        "specialization": "Criminal Defense",
        "experience": "10+ years in extortion cases",
        "emergency_fee": "₹5000-8000"
      }
    ]
  }
}
```

---

## Key Differences in System Handling:

### Consumer Case (Low Priority):
- ✅ Self-help guidance provided
- ✅ Step-by-step process explained
- ✅ No immediate advocate needed
- ✅ Focus on consumer remedies

### Criminal Threat Case (Critical Priority):
- 🚨 Emergency protocol activated
- 🚨 Immediate advocate matching triggered
- 🚨 Strong warnings and urgent language
- 🚨 Anticipatory bail guidance
- 🚨 Evidence preservation priority
- 🚨 Multiple IPC sections referenced

## System Intelligence Features:

1. **Threat Detection**: Automatically identifies "fake rape case" as criminal extortion
2. **Urgency Scaling**: Escalates critical cases to emergency protocols
3. **Context Understanding**: Differentiates between consumer issue vs criminal threat
4. **Legal Accuracy**: Provides correct IPC sections and procedures
5. **Safety First**: Prioritizes user protection over general legal advice
6. **Evidence Focus**: Emphasizes documentation in high-stakes cases

## Implementation Notes:

- **Ollama 3.2B** handles query classification and legal reasoning
- **Gemini** simplifies complex legal language for common users
- **Indian Kanoon** provides case law precedents for similar situations
- **Vector DB** stores and retrieves similar cases for pattern matching
- **Emergency Protocols** activate for threats, extortion, and urgent legal matters