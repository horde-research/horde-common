import logging
from typing import Dict, Any, Callable, List, Optional
from langgraph.graph import StateGraph, END

from .state import PipelineState

logger = logging.getLogger(__name__)


class DataPipelineWorkflow:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, Callable] = {}
        self.edges: List[tuple] = []
        self.entry_point: Optional[str] = None
        self.graph: Optional[StateGraph] = None
    
    def add_node(self, name: str, node_func: Callable[[PipelineState], PipelineState]):
        self.nodes[name] = node_func
        logger.debug("Added node: %s", name)
    
    def add_edge(self, from_node: str, to_node: str):
        self.edges.append((from_node, to_node))
        logger.debug("Added edge: %s -> %s", from_node, to_node)
    
    def set_entry_point(self, node_name: str):
        self.entry_point = node_name
        logger.debug("Set entry point: %s", node_name)
    
    def build(self) -> StateGraph:
        if not self.entry_point:
            raise ValueError("Entry point not set. Call set_entry_point() first.")
        
        if not self.nodes:
            raise ValueError("No nodes added. Add nodes before building.")
        
        workflow = StateGraph(PipelineState)
        
        for name, node_func in self.nodes.items():
            workflow.add_node(name, node_func)
        
        workflow.set_entry_point(self.entry_point)
        
        for from_node, to_node in self.edges:
            if to_node == "END":
                workflow.add_edge(from_node, END)
            else:
                workflow.add_edge(from_node, to_node)
        
        self.graph = workflow.compile()
        logger.info("Workflow built with %d nodes and %d edges", len(self.nodes), len(self.edges))
        return self.graph
    
    def run(self, initial_state: PipelineState) -> Dict[str, Any]:
        if self.graph is None:
            self.build()
        
        if "config" not in initial_state:
            initial_state["config"] = self.config
        
        logger.info("Starting workflow execution")
        final_state = self.graph.invoke(initial_state)
        logger.info("Workflow execution completed")
        return final_state

