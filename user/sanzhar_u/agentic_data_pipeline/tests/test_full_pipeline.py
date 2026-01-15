#!/usr/bin/env python3
"""Test script for full pipeline - connects all agents together."""
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
from src.agents import CategoryAgent, SubcategoryAgent, KeywordAgent


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
    """Test full pipeline: Category -> Subcategory -> Keyword."""
    print("=" * 70)
    print("Full Pipeline Test - All Agents Connected")
    print("=" * 70)
    
    # Initialize LLM and all agents
    try:
        llm = get_llm()
        category_agent = CategoryAgent(llm)
        subcategory_agent = SubcategoryAgent(llm)
        keyword_agent = KeywordAgent(llm)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)
    
    # Test country/culture
    country = "Kazakhstan"
    print(f"\nCountry/Culture: {country}\n")
    
    # Step 1: Generate Categories
    print("=" * 70)
    print("STEP 1: Generating Categories")
    print("=" * 70)
    
    try:
        categories = category_agent.extract_categories(country)
        print(f"\n✓ Generated {len(categories)} categories\n")
        
        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat.get('name', 'Unknown')}")
            print(f"   {cat.get('description', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"✗ Error generating categories: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Generate Subcategories (for ALL categories)
    print("\n" + "=" * 70)
    print("STEP 2: Generating Subcategories")
    print("=" * 70)
    
    category_subcategories = {}
    
    for idx, category in enumerate(categories, 1):
        category_name = category["name"]
        category_desc = category.get("description", "")
        
        print(f"\n[{idx}/{len(categories)}] Category: {category_name}")
        print(f"Description: {category_desc[:80]}...\n")
        
        try:
            subcategories = subcategory_agent.generate_subcategories(
                category_name=category_name,
                category_description=category_desc,
                country_or_culture=country
            )
            
            category_subcategories[category_name] = subcategories
            print(f"✓ Generated {len(subcategories)} subcategories:")
            for i, subcat in enumerate(subcategories, 1):
                print(f"   {i}. {subcat.get('name', 'Unknown')}")
                print(f"      {subcat.get('description', 'N/A')[:80]}...")
            
        except Exception as e:
            print(f"✗ Error generating subcategories: {e}")
            import traceback
            traceback.print_exc()
            category_subcategories[category_name] = []
    
    # Step 3: Generate Keywords (for ALL subcategories of ALL categories)
    print("\n" + "=" * 70)
    print("STEP 3: Generating Keywords")
    print("=" * 70)
    
    category_subcategory_keywords = {}
    
    for category_name, subcategories in category_subcategories.items():
        if not subcategories:
            continue
            
        category_subcategory_keywords[category_name] = {}
        
        # Process ALL subcategories for each category
        for subcat_idx, subcategory in enumerate(subcategories, 1):
            subcategory_name = subcategory["name"]
            subcategory_desc = subcategory.get("description", "")
            
            print(f"\nCategory: {category_name}")
            print(f"[{subcat_idx}/{len(subcategories)}] Subcategory: {subcategory_name}")
            print(f"Description: {subcategory_desc[:80]}...\n")
            
            try:
                keywords = keyword_agent.generate_keywords(
                    category_name=category_name,
                    subcategory_name=subcategory_name,
                    subcategory_description=subcategory_desc,
                    country_or_culture=country
                )
                
                category_subcategory_keywords[category_name][subcategory_name] = keywords
                print(f"✓ Generated {len(keywords)} keywords:")
                for i, keyword in enumerate(keywords[:5], 1):  # Show first 5 keywords
                    print(f"   {i}. {keyword}")
                if len(keywords) > 5:
                    print(f"   ... and {len(keywords) - 5} more keywords")
                
            except Exception as e:
                print(f"✗ Error generating keywords: {e}")
                import traceback
                traceback.print_exc()
                category_subcategory_keywords[category_name][subcategory_name] = []
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"\nCountry/Culture: {country}")
    print(f"Categories generated: {len(categories)}")
    
    total_subcategories = sum(len(subs) for subs in category_subcategories.values())
    print(f"Subcategories generated: {total_subcategories}")
    
    total_keywords = sum(
        len(keywords)
        for subcats in category_subcategory_keywords.values()
        for keywords in subcats.values()
    )
    print(f"Keywords generated: {total_keywords}")
    
    # Check for multilingual keywords
    all_keywords = []
    for subcats in category_subcategory_keywords.values():
        for keywords in subcats.values():
            all_keywords.extend(keywords)
    
    # Simple check for non-ASCII characters (likely native language)
    multilingual_count = sum(1 for kw in all_keywords if any(ord(c) > 127 for c in kw))
    print(f"Multilingual keywords (non-ASCII): {multilingual_count}/{len(all_keywords)}")
    
    # Save results
    results = {
        "country": country,
        "categories": categories,
        "category_subcategories": category_subcategories,
        "category_subcategory_keywords": category_subcategory_keywords,
        "summary": {
            "total_categories": len(categories),
            "total_subcategories": total_subcategories,
            "total_keywords": total_keywords,
            "multilingual_keywords": multilingual_count
        }
    }
    
    output_file = "test_full_pipeline_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Full results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("JSON Output Preview:")
    print("=" * 70)
    print(json.dumps({
        "country": country,
        "categories_count": len(categories),
        "subcategories_count": total_subcategories,
        "keywords_count": total_keywords,
        "sample_keywords": all_keywords[:10] if all_keywords else []
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
