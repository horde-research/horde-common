from typing import TypedDict, List, Dict, Any, Optional


class PipelineState(TypedDict):
    # Input
    free_text: str
    original_context: str
    
    # Category extraction
    categories: List[Dict[str, str]]
    
    # Subcategory generation (nested structure)
    category_subcategories: Dict[str, List[Dict[str, str]]]
    
    # Keyword generation (nested structure: category -> subcategory -> keywords)
    category_subcategory_keywords: Dict[str, Dict[str, List[str]]]
    
    # Collection results (generic - can be images, text, etc.)
    collection_results: List[Dict[str, Any]]
    
    # Current processing state
    current_category: str
    current_subcategory: str
    current_keyword: str
    
    # Configuration (stored in state for access by nodes)
    config: Dict[str, Any]

