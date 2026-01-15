import logging
import os
from ..core.state import PipelineState

logger = logging.getLogger(__name__)


def collect_data_node(state: PipelineState) -> PipelineState:
    logger.info("Starting data collection for all keywords...")
    
    collector = state["config"]["collector"]
    collection_config = state["config"]["collection_config"]
    
    collection_results = []
    
    data_path = collection_config.get("data_path", "./data")
    os.makedirs(data_path, exist_ok=True)
    
    for category in state["categories"]:
        category_name = category["name"]
        category_keywords = state["category_subcategory_keywords"].get(category_name, {})
        
        for subcategory_name, keywords in category_keywords.items():
            for keyword in keywords:
                logger.info(
                    "Collecting data for keyword '%s' (category: %s, subcategory: %s)",
                    keyword,
                    category_name,
                    subcategory_name
                )
                
                keyword_path = os.path.join(
                    data_path,
                    category_name,
                    subcategory_name
                )
                
                try:
                    result = collector.collect(
                        keyword=keyword,
                        output_path=keyword_path,
                        config=collection_config
                    )
                    
                    result.update({
                        "category": category_name,
                        "subcategory": subcategory_name,
                        "keyword": keyword
                    })
                    
                    collection_results.append(result)
                    
                    if result.get("success"):
                        logger.info(
                            "Successfully collected data for keyword '%s'",
                            keyword
                        )
                    else:
                        logger.error(
                            "Failed to collect data for keyword '%s': %s",
                            keyword,
                            result.get("error", "Unknown error")
                        )
                
                except Exception as e:
                    logger.error("Error collecting data for keyword '%s': %s", keyword, e)
                    collection_results.append({
                        "success": False,
                        "category": category_name,
                        "subcategory": subcategory_name,
                        "keyword": keyword,
                        "error": str(e)
                    })
    
    logger.info("Completed data collection. Processed %d keywords", len(collection_results))
    
    return {
        **state,
        "collection_results": collection_results
    }

