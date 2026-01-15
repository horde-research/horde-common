#!/usr/bin/env python3
"""Test script that uses the actual main pipeline code with limited processing."""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

# Import from main.py
from main import create_graph, get_llm
from src.core.state import PipelineState

# Import nodes to create wrapper versions
from src.nodes.category_node import extract_categories_node
from src.nodes.subcategory_node import generate_subcategories_node
from src.nodes.keyword_node import generate_keywords_node


def limited_subcategories_node(state: PipelineState) -> PipelineState:
    """
    Wrapper around generate_subcategories_node that limits to only 1 subcategory per category.
    Note: Categories should already be limited to 3 before calling this.
    """
    # Generate subcategories for all categories
    state = generate_subcategories_node(state)
    
    # Limit to 1 subcategory per category
    limited_subcategories = {}
    for category_name, subcategories in state["category_subcategories"].items():
        if subcategories:
            original_count = len(subcategories)
            limited_subcategories[category_name] = subcategories[:1]
            if original_count > 1:
                print(f"⚠️  Limited to 1 subcategory for '{category_name}' (out of {original_count} total)")
        else:
            limited_subcategories[category_name] = []
    
    return {
        **state,
        "category_subcategories": limited_subcategories
    }


def main():
    """Test the main pipeline with limited processing."""
    print("=" * 70)
    print("Main Pipeline Test - Using Actual Pipeline Code")
    print("Limited: 3 categories, 1 subcategory per category")
    print("=" * 70)
    
    # Test country/culture
    country = "Kazakhstan"
    print(f"\nCountry/Culture: {country}\n")
    
    # Initialize LLM
    try:
        llm = get_llm()
        print("✓ LLM initialized")
    except Exception as e:
        print(f"✗ Error initializing LLM: {e}")
        sys.exit(1)
    
    # Create graph using main.py's create_graph function
    print("\nBuilding pipeline graph...")
    try:
        # Use minimal config for testing (won't actually collect data)
        graph, builder, config = create_graph(
            llm_provider=os.getenv("LLM_PROVIDER", "google"),
            llm_model=os.getenv("LLM_MODEL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            webdriver_path="/usr/local/bin/chromedriver",  # Dummy path for testing
            data_path="./test_data",
            items_per_keyword=1,  # Minimal for testing
            headless=True,
            min_resolution=(100, 100),
            max_resolution=(200, 200),
            max_missed=1
        )
        print("✓ Pipeline graph created")
    except Exception as e:
        print(f"✗ Error creating graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create initial state
    initial_state = builder.create_initial_state(free_text=country)
    print(f"✓ Initial state created with input: {country}")
    
    # Run workflow step by step to allow limiting
    print("\n" + "=" * 70)
    print("STEP 1: Extracting Categories")
    print("=" * 70)
    
    try:
        # Run category extraction
        state_after_categories = extract_categories_node(initial_state)
        
        # Limit to 3 categories
        if len(state_after_categories["categories"]) > 3:
            original_count = len(state_after_categories["categories"])
            state_after_categories["categories"] = state_after_categories["categories"][:3]
            print(f"\n⚠️  Limited to first 3 categories (out of {original_count} total)")
        
        print(f"\n✓ Generated {len(state_after_categories['categories'])} categories:")
        for i, cat in enumerate(state_after_categories["categories"], 1):
            print(f"   {i}. {cat.get('name', 'Unknown')}")
        
    except Exception as e:
        print(f"✗ Error extracting categories: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Generate subcategories (with limiting)
    print("\n" + "=" * 70)
    print("STEP 2: Generating Subcategories (Limited to 1 per category)")
    print("=" * 70)
    
    try:
        state_after_subcategories = limited_subcategories_node(state_after_categories)
        
        total_subcategories = sum(len(subs) for subs in state_after_subcategories["category_subcategories"].values())
        print(f"\n✓ Generated {total_subcategories} subcategories (1 per category):")
        for category_name, subcategories in state_after_subcategories["category_subcategories"].items():
            if subcategories:
                print(f"   • {category_name}:")
                print(f"     - {subcategories[0].get('name', 'Unknown')}")
        
    except Exception as e:
        print(f"✗ Error generating subcategories: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Generate keywords
    print("\n" + "=" * 70)
    print("STEP 3: Generating Keywords")
    print("=" * 70)
    
    try:
        state_after_keywords = generate_keywords_node(state_after_subcategories)
        
        total_keywords = sum(
            len(keywords)
            for subcats in state_after_keywords["category_subcategory_keywords"].values()
            for keywords in subcats.values()
        )
        print(f"\n✓ Generated {total_keywords} keywords:")
        for category_name, subcats in state_after_keywords["category_subcategory_keywords"].items():
            for subcat_name, keywords in subcats.items():
                print(f"   • {category_name} / {subcat_name}:")
                print(f"     {len(keywords)} keywords")
                for i, kw in enumerate(keywords[:5], 1):  # Show first 5
                    print(f"       {i}. {kw}")
                if len(keywords) > 5:
                    print(f"       ... and {len(keywords) - 5} more")
        
    except Exception as e:
        print(f"✗ Error generating keywords: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE TEST SUMMARY")
    print("=" * 70)
    
    final_state = state_after_keywords
    
    print(f"\nCountry/Culture: {country}")
    print(f"Categories generated: {len(final_state['categories'])}")
    
    total_subcategories = sum(len(subs) for subs in final_state["category_subcategories"].values())
    print(f"Subcategories generated: {total_subcategories}")
    
    total_keywords = sum(
        len(keywords)
        for subcats in final_state["category_subcategory_keywords"].values()
        for keywords in subcats.values()
    )
    print(f"Keywords generated: {total_keywords}")
    
    # Check for multilingual keywords
    all_keywords = []
    for subcats in final_state["category_subcategory_keywords"].values():
        for keywords in subcats.values():
            all_keywords.extend(keywords)
    
    multilingual_count = sum(1 for kw in all_keywords if any(ord(c) > 127 for c in kw))
    print(f"Multilingual keywords (non-ASCII): {multilingual_count}/{len(all_keywords)}")
    
    # Save results
    results = {
        "country": country,
        "categories": final_state["categories"],
        "category_subcategories": final_state["category_subcategories"],
        "category_subcategory_keywords": final_state["category_subcategory_keywords"],
        "summary": {
            "total_categories": len(final_state["categories"]),
            "total_subcategories": total_subcategories,
            "total_keywords": total_keywords,
            "multilingual_keywords": multilingual_count
        }
    }
    
    output_file = "test_main_pipeline_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Full results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("✓ Main pipeline test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
