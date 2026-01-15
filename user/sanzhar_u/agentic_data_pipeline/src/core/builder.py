import logging
from typing import Dict, Any, Optional

from .workflow import DataPipelineWorkflow
from .state import PipelineState

logger = logging.getLogger(__name__)


class PipelineBuilder:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow = DataPipelineWorkflow(config)
        self._node_order: list = []
    
    def add_node(self, name: str, node_func, after: Optional[str] = None):
        self.workflow.add_node(name, node_func)
        
        if after:
            self.workflow.add_edge(after, name)
        elif self._node_order:
            self.workflow.add_edge(self._node_order[-1], name)
        
        self._node_order.append(name)
        return self
    
    def set_entry_point(self, node_name: str):
        self.workflow.set_entry_point(node_name)
        return self
    
    def connect_to_end(self, node_name: str):
        self.workflow.add_edge(node_name, "END")
        return self
    
    def build(self) -> DataPipelineWorkflow:
        if not self._node_order:
            raise ValueError("No nodes added to pipeline")
        
        if self._node_order:
            last_node = self._node_order[-1]
            if not any(to == "END" for _, to in self.workflow.edges if _ == last_node):
                self.workflow.add_edge(last_node, "END")
        
        self.workflow.build()
        return self.workflow
    
    def create_initial_state(self, free_text: str, original_context: Optional[str] = None) -> PipelineState:
        return PipelineState(
            free_text=free_text,
            original_context=original_context or free_text,
            categories=[],
            category_subcategories={},
            category_subcategory_keywords={},
            collection_results=[],
            current_category="",
            current_subcategory="",
            current_keyword="",
            config=self.config
        )

