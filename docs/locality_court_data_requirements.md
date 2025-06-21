# Locality-Based Court and Property Valuation Data Requirements

## 1. Geographic Jurisdiction Mapping

### Court Hierarchy and Jurisdiction Data
```json
{
  "jurisdiction_mapping": {
    "state": "Haryana",
    "district": "Gurgaon",
    "sub_district": "Gurgaon-1",
    "pincode": "122001",
    "area_name": "Sector 14, Gurgaon",
    "coordinates": {
      "latitude": 28.4595,
      "longitude": 77.0266
    },
    "court_hierarchy": {
      "civil_court": {
        "junior_civil_judge": "JCJ Court Gurgaon",
        "senior_civil_judge": "SCJ Court Gurgaon", 
        "district_judge": "District Court Gurgaon",
        "additional_district_judge": "ADJ-1 Gurgaon"
      },
      "criminal_court": {
        "judicial_magistrate": "JMIC Gurgaon",
        "chief_judicial_magistrate": "CJM Gurgaon",
        "sessions_judge": "Sessions Court Gurgaon"
      },
      "revenue_court": {
        "tehsildar": "Tehsil Gurgaon",
        "sub_divisional_magistrate": "SDM Gurgaon",
        "district_collector": "DC Gurgaon"
      },
      "specialized_courts": {
        "consumer_forum": "District Consumer Forum Gurgaon",
        "family_court": "Family Court Gurgaon",
        "motor_accident_tribunal": "MACT Gurgaon"
      }
    }
  }
}
```

### Court Selection Rules by Case Type and Value
```json
{
  "court_selection_matrix": {
    "consumer_protection": {
      "up_to_20_lakh": {
        "court": "District Consumer Forum",
        "address": "Mini Secretariat, Sector 17, Gurgaon",
        "filing_fee": "₹100",
        "jurisdiction_rule": "Place of purchase OR consumer residence"
      },
      "20_lakh_to_1_crore": {
        "court": "State Consumer Commission", 
        "address": "Chandigarh",
        "filing_fee": "₹500",
        "jurisdiction_rule": "State where cause of action arose"
      },
      "above_1_crore": {
        "court": "National Consumer Commission",
        "address": "New Delhi", 
        "filing_fee": "₹5000",
        "jurisdiction_rule": "Pan-India jurisdiction"
      }
    },
    "property_disputes": {
      "up_to_3_lakh": {
        "court": "Junior Civil Judge",
        "pecuniary_jurisdiction": "₹3,00,000",
        "court_fee_percentage": "1% of dispute value"
      },
      "3_lakh_to_20_lakh": {
        "court": "Senior Civil Judge", 
        "pecuniary_jurisdiction": "₹20,00,000",
        "court_fee_percentage": "1% of dispute value"
      },
      "above_20_lakh": {
        "court": "District Judge",
        "pecuniary_jurisdiction": "Unlimited",
        "court_fee_percentage": "1% up to ₹1 crore, then 0.5%"
      }
    },
    "criminal_cases": {
      "punishable_up_to_3_years": {
        "court": "Judicial Magistrate",
        "examples": ["Simple hurt", "Cheating", "Criminal breach of trust"]
      },
      "punishable_above_3_years": {
        "court": "Sessions Court",
        "examples": ["Murder", "Rape", "Dacoity"]
      }
    }
  }
}
```

## 2. Property Valuation and Registration Data

### Circle Rate and Valuation Matrix
```json
{
  "property_valuation_data": {
    "gurgaon_sector_14": {
      "circle_rate_per_sqyd": {
        "residential": 85000,
        "commercial": 125000,
        "industrial": 65000
      },
      "market_rate_per_sqyd": {
        "residential": 95000,
        "commercial": 140000,
        "industrial": 70000
      },
      "registration_office": "Sub-Registrar Office, Sector 17, Gurgaon",
      "stamp_duty_rates": {
        "male_buyer": "7%",
        "female_buyer": "6%", 
        "joint_ownership": "6%"
      },
      "registration_fee": "1% of property value",
      "minimum_charges": {
        "stamp_duty": "₹100",
        "registration_fee": "₹60"
      }
    }
  }
}
```

### Registration Jurisdiction Rules
```json
{
  "registration_jurisdiction": {
    "rule": "Property location determines jurisdiction",
    "gurgaon_sectors": {
      "sectors_1_to_15": "Sub-Registrar Office Sector 17",
      "sectors_16_to_30": "Sub-Registrar Office Sector 31", 
      "sectors_31_to_50": "Sub-Registrar Office Sector 45",
      "dlf_phases": "Sub-Registrar Office DLF"
    },
    "required_documents": [
      "Sale deed",
      "NOC from society/builder", 
      "Property tax clearance",
      "Encumbrance certificate",
      "Identity and address proof of all parties"
    ]
  }
}
```

## 3. Local Court Infrastructure Data

### Court Contact and Timing Information
```json
{
  "court_infrastructure": {
    "district_court_gurgaon": {
      "address": "District Court Complex, Sector 51, Gurgaon",
      "contact": {
        "phone": "0124-2345678",
        "email": "dcgurgaon@hry.nic.in",
        "website": "districts.ecourts.gov.in/gurgaon"
      },
      "working_hours": {
        "court_hours": "10:00 AM - 5:00 PM",
        "filing_hours": "10:00 AM - 4:00 PM",
        "holidays": "All Sundays and gazetted holidays"
      },
      "facilities": {
        "parking": true,
        "cafeteria": true,
        "lawyer_chambers": true,
        "digital_filing": true,
        "e_payment": true
      },
      "case_types_handled": [
        "Civil suits",
        "Criminal cases", 
        "Family disputes",
        "Property disputes",
        "Motor accident claims"
      ]
    }
  }
}
```

### Judge Allocation and Court Capacity
```json
{
  "judge_allocation": {
    "district_court_gurgaon": {
      "total_judges": 15,
      "civil_judges": 8,
      "criminal_judges": 5,
      "family_court_judges": 2,
      "average_case_load": 450,
      "average_disposal_time": {
        "civil_cases": "18 months",
        "criminal_cases": "12 months", 
        "family_cases": "24 months"
      },
      "court_language": "Hindi/English",
      "interpreter_available": true
    }
  }
}
```

## 4. Local Legal Procedure Variations

### State-Specific Rules and Procedures
```json
{
  "haryana_specific_rules": {
    "property_registration": {
      "mandatory_documents": [
        "Haryana Urban Development Authority (HUDA) clearance",
        "Fire safety certificate",
        "Pollution clearance certificate"
      ],
      "additional_fees": {
        "conversion_charges": "₹50,000 (agricultural to residential)",
        "external_development_charges": "₹500 per sqyd"
      },
      "time_limits": {
        "registration_completion": "4 months from agreement",
        "stamp_duty_payment": "Within 30 days"
      }
    },
    "consumer_complaints": {
      "local_provisions": [
        "Haryana Consumer Protection Rules 2021",
        "E-filing mandatory for claims above ₹10 lakh"
      ],
      "mediation_centers": [
        "Lok Adalat Gurgaon - Every 2nd Saturday",
        "Consumer Mediation Center - Sector 17"
      ]
    }
  }
}
```

## 5. Transport and Accessibility Data

### Court Accessibility Information
```json
{
  "accessibility_data": {
    "transport_options": {
      "nearest_metro": "Huda City Centre (2.5 km)",
      "bus_routes": ["Route 5", "Route 12A", "Route 23"],
      "auto_fare": "₹150-200 from Gurgaon Railway Station",
      "parking_availability": "200 vehicle capacity"
    },
    "nearby_facilities": {
      "lawyer_chambers": "50+ advocates within 500m",
      "document_services": [
        "Xerox shops",
        "Notary services", 
        "Typing centers"
      ],
      "banks_atm": "3 banks within 1km for fee payment"
    }
  }
}
```

## 6. Historical Case Data and Trends

### Local Court Performance Analytics
```json
{
  "court_analytics": {
    "case_disposal_trends": {
      "2023_data": {
        "total_cases_filed": 12450,
        "cases_disposed": 10200,
        "disposal_rate": "82%",
        "average_disposal_time": {
          "property_disputes": "16 months",
          "consumer_complaints": "8 months",
          "criminal_cases": "14 months"
        }
      }
    },
    "judge_wise_performance": {
      "civil_judge_1": {
        "disposal_rate": "85%",
        "average_hearing_gap": "21 days",
        "specialization": "Property disputes"
      }
    }
  }
}
```

## 7. Fee Calculation and Payment Systems

### Dynamic Fee Calculator Data
```json
{
  "fee_calculator": {
    "property_registration": {
      "base_calculation": {
        "circle_rate": "auto_fetch_by_location",
        "agreement_value": "user_input",
        "applicable_rate": "higher_of_both",
        "stamp_duty": "rate * applicable_value",
        "registration_fee": "1% of applicable_value"
      },
      "additional_charges": {
        "scanning_fee": "₹50",
        "certified_copy": "₹20 per page",
        "urgent_registration": "₹1000 extra"
      }
    },
    "court_fees": {
      "civil_suit": {
        "calculation": "1% of suit value",
        "minimum": "₹15",
        "maximum": "₹7500"
      }
    }
  }
}
```

## 8. Real-Time Court Status

### Live Court Information
```json
{
  "real_time_status": {
    "court_functioning": {
      "status": "Open/Closed/Holiday",
      "current_board": "Board number being heard",
      "expected_delay": "30 minutes",
      "lunch_break": "13:00-14:00"
    },
    "case_status_tracking": {
      "next_hearing_date": "auto_update",
      "case_stage": "evidence/argument/judgment",
      "judge_assigned": "current_judge_name"
    }
  }
}
```

## 9. Integration with Government APIs

### API Connections Required
```python
government_apis = {
    "e_courts": {
        "url": "https://ecourts.gov.in/ecourts_home/",
        "data": "Case status, hearing dates, judgments"
    },
    "registration_department": {
        "url": "https://stamps.nic.in/",
        "data": "Circle rates, stamp duty calculation"
    },
    "revenue_records": {
        "url": "https://jamabandi.nic.in/",
        "data": "Property ownership records"
    },
    "consumer_forum": {
        "url": "https://edaakhil.nic.in/",
        "data": "Consumer complaint status"
    }
}
```

## 10. Machine Learning Training Data

### Training Data Requirements
```python
ml_training_data = {
    "court_recommendation": {
        "features": [
            "case_type",
            "dispute_value", 
            "user_location",
            "property_location",
            "case_urgency",
            "preferred_language"
        ],
        "labels": "recommended_court_with_reasons"
    },
    "property_valuation": {
        "features": [
            "location_coordinates",
            "property_type",
            "area_sqft",
            "age_of_property",
            "amenities"
        ],
        "labels": "estimated_market_value"
    },
    "case_duration_prediction": {
        "features": [
            "court_name",
            "case_type",
            "judge_assigned",
            "complexity_score"
        ],
        "labels": "expected_disposal_time"
    }
}
```

This comprehensive data structure will enable your LegalLink system to provide accurate, locality-specific legal guidance with proper court recommendations based on case type, property valuation, and jurisdictional requirements.