import logging
import time
import os
from ..core.state import PipelineState
from ..agents import SubcategoryAgent

logger = logging.getLogger(__name__)

API_DELAY_SECONDS = float(os.getenv("API_DELAY_SECONDS", "1.5"))


def generate_subcategories_node(state: PipelineState) -> PipelineState:
    logger.info("Generating subcategories for %d categories...", len(state["categories"]))
    
    subcategory_agent: SubcategoryAgent = state["config"]["agents"]["subcategory_agent"]
    
    category_subcategories = {}
    country_or_culture = state.get("original_context", state["free_text"])
    
    for idx, category in enumerate(state["categories"], 1):
        category_name = category["name"]
        category_desc = category.get("description", "")
        
        if idx > 1:
            time.sleep(API_DELAY_SECONDS)
        
        subcategories = subcategory_agent.generate_subcategories(
            category_name=category_name,
            category_description=category_desc,
            country_or_culture=country_or_culture
        )
        
        category_subcategories[category_name] = subcategories
        logger.info("Generated %d subcategories for category '%s' (%d/%d)", 
                   len(subcategories), category_name, idx, len(state["categories"]))
    
    return {
        **state,
        "category_subcategories": category_subcategories
    }

