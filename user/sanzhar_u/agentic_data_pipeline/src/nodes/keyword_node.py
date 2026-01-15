"""Node for generating keywords."""
import logging
import time
import os
from ..core.state import PipelineState
from ..agents import KeywordAgent

logger = logging.getLogger(__name__)

API_DELAY_SECONDS = float(os.getenv("API_DELAY_SECONDS", "1.5"))


def generate_keywords_node(state: PipelineState) -> PipelineState:
    """
    Generate keywords for all subcategories.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with keywords
    """
    logger.info("Generating keywords for all subcategories...")
    
    keyword_agent: KeywordAgent = state["config"]["agents"]["keyword_agent"]
    
    category_subcategory_keywords = {}
    total_subcategories = sum(len(subs) for subs in state["category_subcategories"].values())
    processed_count = 0
    
    for category in state["categories"]:
        category_name = category["name"]
        category_subcategory_keywords[category_name] = {}
        
        subcategories = state["category_subcategories"].get(category_name, [])
        
        for subcategory in subcategories:
            subcategory_name = subcategory["name"]
            subcategory_desc = subcategory.get("description", "")
            
            processed_count += 1
            if processed_count > 1:
                time.sleep(API_DELAY_SECONDS)
            
            country_or_culture = state.get("original_context", state["free_text"])
            
            keywords = keyword_agent.generate_keywords(
                category_name=category_name,
                subcategory_name=subcategory_name,
                subcategory_description=subcategory_desc,
                country_or_culture=country_or_culture
            )
            
            category_subcategory_keywords[category_name][subcategory_name] = keywords
            logger.info(
                "Generated %d keywords for subcategory '%s/%s' (%d/%d)",
                len(keywords),
                category_name,
                subcategory_name,
                processed_count,
                total_subcategories
            )
    
    return {
        **state,
        "category_subcategory_keywords": category_subcategory_keywords
    }

