import logging
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SubcategoryAgent(BaseAgent):
    """Agent that generates comprehensive subcategories for cultural categories."""
    
    SYSTEM_PROMPT = """You are an expert cultural analyst specializing in breaking down broad cultural categories into specific, meaningful, and comprehensive subcategories.

Your task is to analyze a cultural category within the context of a specific country or culture and create a detailed, professional list of subcategories that would help organize data collections more granularly and comprehensively.

Output a JSON object with the following structure:
{{
    "subcategories": [
        {{
            "name": "subcategory_name_in_english",
            "description": "detailed, comprehensive description of what this subcategory represents, what types of data it encompasses, and its significance within the cultural context"
        }}
    ]
}}

When generating subcategories, you must:

1. **Understand the Cultural Context**:
   - Consider the specific country or culture being documented
   - Account for unique cultural characteristics, traditions, and practices
   - Reflect authentic cultural expressions and variations

2. **Be Comprehensive and Detailed**:
   - Cover all major aspects and variations of the category
   - Include both traditional and contemporary elements
   - Consider regional, social, and temporal variations
   - Think about different perspectives and use cases

3. **Ensure Professional Quality**:
   - Use precise, descriptive language
   - Make subcategories specific enough to be actionable
   - Ensure subcategories are distinct and non-overlapping
   - Consider practical data collection needs

4. **Consider the Category Description**:
   - Use the category description to understand the scope
   - Ensure subcategories align with the category's purpose
   - Create subcategories that logically fit within the category

Generate 4-10 comprehensive, well-described subcategories per category. Each subcategory should be specific, meaningful, and useful for organizing cultural data collection.

Return only valid JSON."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, self.SYSTEM_PROMPT)
    
    def generate_subcategories(
        self, 
        category_name: str, 
        category_description: str,
        country_or_culture: str = ""
    ) -> List[Dict[str, str]]:
        country_context = f"\n\nCountry/Culture Context: {country_or_culture}" if country_or_culture else ""
        
        user_message = f"""Generate comprehensive subcategories for the following cultural category:

Category: {category_name}
Category Description: {category_description}{country_context}

Create detailed, professional subcategories that:
- Are specific to the cultural context provided
- Cover all major aspects and variations of this category
- Include both traditional and contemporary elements
- Are well-described and actionable for data collection
- Reflect authentic cultural expressions

Provide a thorough breakdown that would enable comprehensive data collection for this category within the specified cultural context."""
        
        try:
            result = self.invoke(user_message)
            subcategories = result.get("subcategories", [])
            logger.info("Generated %d subcategories for category '%s'", len(subcategories), category_name)
            return subcategories
        except Exception as e:
            logger.error("Error generating subcategories: %s", e)
            return []

