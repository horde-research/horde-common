#!/usr/bin/env python3
"""Test script for KeywordAgent - generates keywords for preset category and subcategory."""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents import KeywordAgent


def get_llm():
    """Get LLM instance from environment."""
    provider = os.getenv("LLM_PROVIDER", "google")
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("LLM_API_KEY")
    
    if provider.lower() == "openai":
        model = model or "gpt-4o-mini"
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.7)
    
    elif provider.lower() == "anthropic":
        model = model or "claude-3-haiku-20240307"
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0.7)
    
    elif provider.lower() == "google":
        model = model or "gemini-2.5-flash"
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.7)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def main():
    """Test KeywordAgent with preset category and subcategory."""
    print("=" * 60)
    print("Keyword Agent Test")
    print("=" * 60)
    
    # Initialize LLM and agent
    try:
        llm = get_llm()
        agent = KeywordAgent(llm)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)
    
    # Preset category and subcategory for testing
    test_category = {
        "name": "Traditional Customs & Practices",
        "description": "Traditional ceremonies, rituals, celebrations, folk traditions, and cultural practices"
    }
    
    test_subcategories = [
        {
            "name": "Traditional Ceremonies",
            "description": "Wedding ceremonies, birth celebrations, coming-of-age rituals, and other significant life events"
        },
        {
            "name": "Folk Festivals",
            "description": "Traditional festivals, seasonal celebrations, and community gatherings"
        },
        {
            "name": "Religious Practices",
            "description": "Religious rituals, spiritual practices, and observances"
        }
    ]
    
    print(f"\nCategory: {test_category['name']}")
    print(f"Description: {test_category['description']}\n")
    
    country_or_culture = "Kazakhstan"
    all_results = {}
    
    for subcategory in test_subcategories:
        print("-" * 60)
        print(f"\nSubcategory: {subcategory['name']}")
        print(f"Description: {subcategory['description']}\n")
        
        try:
            keywords = agent.generate_keywords(
                category_name=test_category["name"],
                subcategory_name=subcategory["name"],
                subcategory_description=subcategory["description"],
                country_or_culture=country_or_culture
            )
            
            print(f"Generated {len(keywords)} keywords:\n")
            for i, keyword in enumerate(keywords, 1):
                print(f"  {i}. {keyword}")
            
            all_results[subcategory["name"]] = {
                "subcategory": subcategory,
                "keywords": keywords
            }
            
        except Exception as e:
            print(f"Error generating keywords: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("JSON Output:")
    print("=" * 60)
    print(json.dumps({
        "category": test_category,
        "subcategory_keywords": all_results
    }, indent=2, ensure_ascii=False))
    
    # Save to file
    output_file = "test_keyword_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": test_category,
            "subcategory_keywords": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
