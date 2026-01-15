#!/usr/bin/env python3
"""Test script for SubcategoryAgent - generates subcategories for preset categories."""
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
from src.agents import SubcategoryAgent


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
    """Test SubcategoryAgent with preset categories."""
    print("=" * 60)
    print("Subcategory Agent Test")
    print("=" * 60)
    
    # Initialize LLM and agent
    try:
        llm = get_llm()
        agent = SubcategoryAgent(llm)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)
    
    # Preset categories for testing
    test_categories = [
        {
            "name": "Traditional Customs & Practices",
            "description": "Traditional ceremonies, rituals, celebrations, folk traditions, and cultural practices"
        },
        {
            "name": "Cuisine & Food Culture",
            "description": "Traditional dishes, food preparation methods, dining customs, and culinary traditions"
        },
        {
            "name": "Cultural Arts & Expression",
            "description": "Traditional music, dance, visual arts, crafts, literature, and contemporary artistic expressions"
        }
    ]
    
    print(f"\nTesting with {len(test_categories)} preset categories:\n")
    
    all_results = {}
    country_or_culture = "Kazakhstan"
    
    for category in test_categories:
        print("-" * 60)
        print(f"\nCategory: {category['name']}")
        print(f"Description: {category['description']}\n")
        
        try:
            subcategories = agent.generate_subcategories(
                category_name=category["name"],
                category_description=category["description"],
                country_or_culture=country_or_culture
            )
            
            print(f"Generated {len(subcategories)} subcategories:\n")
            for i, subcat in enumerate(subcategories, 1):
                print(f"  {i}. {subcat.get('name', 'Unknown')}")
                print(f"     Description: {subcat.get('description', 'N/A')}")
            
            all_results[category["name"]] = {
                "category": category,
                "subcategories": subcategories
            }
            
        except Exception as e:
            print(f"Error generating subcategories: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("JSON Output:")
    print("=" * 60)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))
    
    # Save to file
    output_file = "test_subcategory_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
