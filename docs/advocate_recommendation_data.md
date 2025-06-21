# Advocate Data Requirements for LegalLink Recommendation System

## Core Advocate Data Structure

### 1. Basic Professional Information
```json
{
  "advocate_id": "ADV_2024_DL_1234",
  "full_name": "Advocate Rajesh Kumar Singh",
  "gender": "Male",
  "age": 42,
  "profile_photo": "url_to_verified_photo",
  "languages": ["Hindi", "English", "Punjabi"],
  "bar_council_details": {
    "registration_number": "D/1234/2008",
    "bar_council": "Bar Council of Delhi",
    "enrollment_date": "2008-07-15",
    "current_status": "Active",
    "last_verified": "2024-01-15"
  }
}
```

### 2. Specialization and Expertise
```json
{
  "specializations": [
    {
      "area": "Consumer Protection Law",
      "experience_years": 12,
      "proficiency_level": "Expert",
      "sub_specializations": [
        "E-commerce disputes",
        "Banking complaints", 
        "Insurance claims",
        "Product liability"
      ]
    },
    {
      "area": "Criminal Law",
      "experience_years": 8,
      "proficiency_level": "Advanced",
      "sub_specializations": [
        "Domestic violence",
        "Cyber crimes",
        "Economic offenses"
      ]
    }
  ],
  "case_types_handled": [
    "Consumer complaints",
    "Criminal defense",
    "Bail applications",
    "Quashing petitions"
  ]
}
```

### 3. Court Practice and Jurisdiction
```json
{
  "practice_courts": [
    {
      "court_name": "Delhi District Court",
      "court_type": "District",
      "practice_years": 15,
      "frequency": "Daily"
    },
    {
      "court_name": "Delhi High Court",
      "court_type": "High Court", 
      "practice_years": 8,
      "frequency": "Weekly"
    }
  ],
  "geographic_coverage": {
    "primary_city": "New Delhi",
    "coverage_areas": ["Delhi NCR", "Gurgaon", "Noida", "Faridabad"],
    "willing_to_travel": true,
    "travel_radius_km": 50
  }
}
```

### 4. Experience and Track Record
```json
{
  "experience_metrics": {
    "total_years_practice": 15,
    "total_cases_handled": 1247,
    "cases_won": 892,
    "cases_settled": 234,
    "success_rate": 89.2,
    "average_case_duration_days": 125
  },
  "notable_cases": [
    {
      "case_type": "Consumer Protection",
      "description": "Landmark e-commerce refund case",
      "year": 2023,
      "outcome": "Won ₹5 lakh compensation for client"
    }
  ],
  "achievements": [
    "Best Consumer Lawyer Award 2023",
    "Published 15 articles on consumer rights"
  ]
}
```

### 5. Availability and Accessibility
```json
{
  "availability": {
    "consultation_modes": ["In-person", "Video call", "Phone call"],
    "working_hours": {
      "monday_friday": "09:00-18:00",
      "saturday": "09:00-14:00", 
      "sunday": "Emergency only"
    },
    "emergency_availability": true,
    "same_day_consultation": true,
    "average_response_time_hours": 2,
    "next_available_slot": "2024-06-21T15:00:00Z"
  },
  "communication_preferences": {
    "preferred_language": "Hindi",
    "communication_style": "Patient and explanatory",
    "client_updates_frequency": "Weekly"
  }
}
```

### 6. Fee Structure and Pricing
```json
{
  "fee_structure": {
    "consultation_fee": {
      "first_consultation": 2000,
      "follow_up_consultation": 1500,
      "phone_consultation": 1000,
      "emergency_consultation": 3000
    },
    "case_fees": {
      "consumer_complaint": "₹15,000 - ₹25,000",
      "criminal_defense": "₹30,000 - ₹75,000",
      "bail_application": "₹20,000 - ₹35,000"
    },
    "payment_options": ["Cash", "UPI", "Bank transfer", "EMI available"],
    "advance_payment": "30% of total fee",
    "no_win_no_fee": false,
    "legal_aid_cases": true
  }
}
```

### 7. Client Reviews and Ratings
```json
{
  "ratings": {
    "overall_rating": 4.8,
    "total_reviews": 127,
    "rating_breakdown": {
      "5_star": 89,
      "4_star": 28,
      "3_star": 8,
      "2_star": 2,
      "1_star": 0
    }
  },
  "client_testimonials": [
    {
      "client_initials": "P.S.",
      "case_type": "Consumer Protection",
      "rating": 5,
      "review": "Excellent lawyer, got my refund within 30 days",
      "date": "2024-05-15",
      "verified": true
    }
  ],
  "professional_endorsements": [
    {
      "endorser": "Senior Advocate M.K. Sharma",
      "relationship": "Senior colleague",
      "endorsement": "Very thorough and client-focused"
    }
  ]
}
```

### 8. Professional Credentials and Verification
```json
{
  "credentials": {
    "law_degree": {
      "degree": "LLB",
      "university": "Delhi University",
      "year": 2007,
      "grade": "First Class"
    },
    "additional_qualifications": [
      "Diploma in Cyber Law (2015)",
      "Certificate in Consumer Protection (2020)"
    ],
    "professional_memberships": [
      "Delhi Bar Association",
      "Consumer Lawyers Association of India"
    ]
  },
  "verification_status": {
    "bar_council_verified": true,
    "address_verified": true,
    "phone_verified": true,
    "email_verified": true,
    "background_check": "Clear",
    "last_verification_date": "2024-01-15"
  }
}
```

### 9. Digital Presence and Accessibility
```json
{
  "digital_profile": {
    "website": "https://advocaterajesh.com",
    "linkedin": "linkedin.com/in/advocaterajesh",
    "professional_blog": "https://legaladvice.rajesh.com",
    "video_consultation_platform": "Zoom, Google Meet",
    "document_sharing": "WhatsApp, Email, Portal",
    "online_payment": true
  },
  "tech_comfort_level": "High",
  "digital_document_handling": true
}
```

### 10. Case Management and Communication
```json
{
  "case_management": {
    "case_tracking_system": true,
    "client_portal_access": true,
    "regular_updates": "Weekly via WhatsApp",
    "document_management": "Digital + Physical",
    "deadline_management": "Automated reminders",
    "court_date_notifications": "48 hours advance"
  },
  "client_support": {
    "paralegal_support": true,
    "junior_associate": "Advocate Priya Sharma",
    "office_staff": 3,
    "multilingual_support": true
  }
}
```

## Recommendation Algorithm Factors

### 1. Matching Criteria (Weighted)
```python
recommendation_weights = {
    "specialization_match": 30,      # Most important
    "geographic_proximity": 20,      # Location convenience
    "experience_level": 15,          # Years + success rate
    "client_ratings": 15,            # Reviews and ratings
    "availability": 10,              # Can take case soon
    "fee_affordability": 5,          # Within user budget
    "language_match": 3,             # Communication comfort
    "court_familiarity": 2           # Practices in relevant court
}
```

### 2. Priority Filters
```python
priority_filters = {
    "emergency_cases": {
        "max_response_time": 2,      # Hours
        "emergency_availability": True,
        "same_day_consultation": True
    },
    "budget_constraints": {
        "max_consultation_fee": 3000,
        "payment_flexibility": True,
        "legal_aid_availability": True
    },
    "case_complexity": {
        "min_experience_years": 5,
        "similar_case_success": True,
        "court_practice_years": 3
    }
}
```

### 3. Red Flag Detection
```python
red_flags = {
    "verification_issues": [
        "Bar council registration expired",
        "Multiple client complaints unresolved",
        "Disciplinary action by bar council"
    ],
    "service_issues": [
        "Consistently poor ratings",
        "Non-responsive to clients",
        "Excessive fee demands"
    ],
    "professional_conduct": [
        "Ethics violations",
        "Conflict of interest issues",
        "Unprofessional behavior reports"
    ]
}
```

## Data Collection Sources

### 1. Official Sources
- **Bar Council of India** - Registration verification
- **State Bar Councils** - Local advocate lists
- **Court websites** - Practice records
- **Legal directories** - Professional listings

### 2. Professional Networks
- **Bar associations** - Member directories
- **Legal chambers** - Advocate profiles
- **Law firm websites** - Attorney information
- **Professional legal platforms** - Verified profiles

### 3. Client Feedback Systems
- **LegalLink platform reviews** - Direct client feedback
- **Third-party legal platforms** - Cross-platform ratings
- **Social media presence** - Professional reputation
- **Peer recommendations** - Professional networks

### 4. Verification Processes
- **Document verification** - Degrees, certificates
- **Court record checks** - Case history
- **Client reference checks** - Past client satisfaction
- **Background verification** - Professional conduct

## Real-time Data Updates

### 1. Dynamic Information
```python
real_time_updates = {
    "availability_calendar": "Updated every 15 minutes",
    "case_load_status": "Updated daily",
    "fee_structure_changes": "Updated monthly",
    "rating_changes": "Updated after each review",
    "court_schedule": "Updated weekly"
}
```

### 2. Automated Quality Checks
```python
quality_monitoring = {
    "response_time_tracking": True,
    "client_satisfaction_surveys": True,
    "case_outcome_tracking": True,
    "fee_dispute_monitoring": True,
    "professional_conduct_alerts": True
}
```

This comprehensive advocate data structure ensures accurate, relevant, and safe recommendations for users seeking legal help through your LegalLink platform.