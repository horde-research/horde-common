import logging
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CategoryAgent(BaseAgent):
    
    SYSTEM_PROMPT = """You are an expert cultural anthropologist and data analyst specializing in comprehensive cultural documentation and analysis.

Your task is to generate detailed, comprehensive categories for documenting and understanding a specific culture, country, or region. You should create categories that capture the full spectrum of cultural expression, social practices, traditions, and contemporary trends.

Output a JSON object with the following structure:
{{
    "categories": [
        {{
            "name": "category_name_in_english",
            "description": "detailed, comprehensive description of what this category represents and what types of data it encompasses"
        }}
    ]
}}

You must be thorough and comprehensive. Generate categories that cover:

1. **Traditional Customs & Practices**:
   - Traditional ceremonies, rituals, and celebrations
   - Folk traditions and cultural practices
   - Religious and spiritual customs
   - Traditional crafts and artisanal work
   - Historical cultural practices

2. **Social Structures & Daily Life**:
   - Family structures and relationships
   - Social hierarchies and community organization
   - Daily routines and lifestyle patterns
   - Social gatherings and community events
   - Interpersonal relationships and social norms

3. **Cultural Arts & Expression**:
   - Traditional music, dance, and performing arts
   - Visual arts, crafts, and traditional design
   - Literature, poetry, and oral traditions
   - Cultural symbols, motifs, and iconography
   - Contemporary artistic expressions

4. **Cuisine & Food Culture**:
   - Traditional dishes and recipes
   - Food preparation methods and techniques
   - Dining customs and etiquette
   - Food-related celebrations and traditions
   - Regional culinary variations

5. **Clothing & Fashion**:
   - Traditional attire and costumes
   - Cultural significance of clothing
   - Contemporary fashion trends
   - Textiles and fabric traditions
   - Accessories and adornments

6. **Architecture & Living Spaces**:
   - Traditional architecture and building styles
   - Interior design and home organization
   - Urban and rural living environments
   - Cultural significance of spaces
   - Modern architectural trends

7. **Language & Communication**:
   - Language use and dialects
   - Communication styles and etiquette
   - Written and oral traditions
   - Contemporary language trends
   - Multilingual aspects

8. **Recent Trends & Modern Culture**:
   - Contemporary social movements
   - Modern lifestyle trends
   - Technology integration in culture
   - Youth culture and generational differences
   - Globalization influences

9. **Geographical & Environmental Context**:
   - Landscape and natural environment
   - Urban vs rural distinctions
   - Regional variations within the culture
   - Environmental practices and traditions
   - Seasonal cultural variations

10. **Economic & Professional Life**:
    - Traditional occupations and trades
    - Modern professional environments
    - Economic activities and markets
    - Work-life balance and practices
    - Entrepreneurship and innovation

Generate 8-15 comprehensive categories that together provide a complete picture of the culture. Each category should be specific enough to be actionable for data collection, yet broad enough to encompass related sub-topics.

Return only valid JSON."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, self.SYSTEM_PROMPT)
    
    def extract_categories(self, country_or_culture: str) -> List[Dict[str, str]]:
        user_message = f"""Generate comprehensive data categories for documenting and understanding the culture, customs, traditions, and recent trends of: {country_or_culture}

Create detailed categories that would enable comprehensive data collection covering:
- Traditional customs and practices
- Social structures and daily life
- Cultural arts and expression
- Cuisine and food culture
- Clothing and fashion
- Architecture and living spaces
- Language and communication
- Recent trends and modern culture
- Geographical and environmental context
- Economic and professional life

Provide a thorough, well-organized set of categories that capture the full spectrum of this culture."""
        
        try:
            result = self.invoke(user_message)
            categories = result.get("categories", [])
            logger.info("Extracted %d categories for %s", len(categories), country_or_culture)
            return categories
        except Exception as e:
            logger.error("Error extracting categories: %s", e)
            return []

