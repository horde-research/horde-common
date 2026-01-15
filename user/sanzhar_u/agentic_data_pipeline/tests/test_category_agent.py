#!/usr/bin/env python3
"""Test script for CategoryAgent - generates categories for a country/culture."""
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
from src.agents import CategoryAgent


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
    """Test CategoryAgent with Kazakhstan."""
    print("=" * 60)
    print("Category Agent Test - Kazakhstan")
    print("=" * 60)
    
    # Initialize LLM and agent
    try:
        llm = get_llm()
        agent = CategoryAgent(llm)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)
    
    # Test with Kazakhstan
    country = "Kazakhstan"
    print(f"\nGenerating categories for: {country}\n")
    
    try:
        categories = agent.extract_categories(country)
        
        print(f"\nGenerated {len(categories)} categories:\n")
        print("-" * 60)
        
        for i, cat in enumerate(categories, 1):
            print(f"\n{i}. {cat.get('name', 'Unknown')}")
            print(f"   Description: {cat.get('description', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("JSON Output:")
        print("=" * 60)
        print(json.dumps({"categories": categories}, indent=2, ensure_ascii=False))
        
        # Save to file
        output_file = "test_category_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"country": country, "categories": categories}, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error generating categories: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
