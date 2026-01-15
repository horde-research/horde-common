import logging
from ..core.state import PipelineState
from ..agents import CategoryAgent

logger = logging.getLogger(__name__)


def extract_categories_node(state: PipelineState) -> PipelineState:
    logger.info("Extracting categories from input...")
    
    category_agent: CategoryAgent = state["config"]["agents"]["category_agent"]
    
    categories = category_agent.extract_categories(state["free_text"])
    
    return {
        **state,
        "categories": categories,
        "category_subcategories": {},
        "category_subcategory_keywords": {},
        "collection_results": []
    }

