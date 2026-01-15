import os
import logging
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel

from ..agents import CategoryAgent, SubcategoryAgent, KeywordAgent
from ..collectors import BaseCollector

logger = logging.getLogger(__name__)


class PipelineConfig:
    def __init__(
        self,
        llm: BaseChatModel,
        collector: BaseCollector,
        collection_config: Dict[str, Any],
        **kwargs
    ):
        self.llm = llm
        self.collector = collector
        self.collection_config = collection_config
        self.extra_config = kwargs
        
        # Initialize agents
        self.agents = {
            "category_agent": CategoryAgent(llm),
            "subcategory_agent": SubcategoryAgent(llm),
            "keyword_agent": KeywordAgent(llm)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": self.agents,
            "collector": self.collector,
            "collection_config": self.collection_config,
            **self.extra_config
        }


def create_image_collection_config(
    llm: BaseChatModel,
    webdriver_path: str,
    data_path: str = "./data",
    number_of_images: int = 30,
    headless: bool = True,
    min_resolution: tuple = (400, 400),
    max_resolution: tuple = (8000, 8000),
    max_missed: int = 200
) -> PipelineConfig:
    from ..collectors import ImageCollector
    
    collector = ImageCollector()
    
    collection_config = {
        "data_path": data_path,
        "webdriver_path": webdriver_path,
        "number_of_images": number_of_images,
        "headless": headless,
        "min_resolution": min_resolution,
        "max_resolution": max_resolution,
        "max_missed": max_missed
    }
    
    return PipelineConfig(
        llm=llm,
        collector=collector,
        collection_config=collection_config
    )

