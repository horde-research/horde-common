import logging
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class KeywordAgent(BaseAgent):
    SYSTEM_PROMPT = """You are an expert at creating effective search keywords for internet search engines (Google, Bing, etc.) and data collection.

Your task is to generate specific, searchable keywords that would return relevant data (images, text, etc.) for a given subcategory within a cultural context.

Output a JSON object with the following structure:
{{
    "keywords": [
        "search keyword phrase 1",
        "search keyword phrase 2",
        "search keyword phrase 3"
    ]
}}

Guidelines for creating effective keywords:

1. **Search Engine Optimization**:
   - Use 2-5 word phrases that people actually search for
   - Include the country/culture name when relevant
   - Use specific, descriptive terms
   - Think about what would return the best results on Google Images or web search

2. **Multilingual Keywords (REQUIRED)**:
   - **ALWAYS generate keywords in BOTH native language(s) AND English**
   - For each country/culture, include keywords in:
     * Native language(s) of the country (e.g., for Kazakhstan: Kazakh language using Cyrillic script like "Қазақстан дәстүрлі", "қазақ киімі")
     * English translations and descriptions (e.g., "Kazakhstan traditional", "Kazakh culture")
   - If the country has multiple official languages, include keywords in all major languages
   - Use native script (Cyrillic, Arabic, Latin, etc.) as appropriate for the country
   - Include transliterations when helpful (e.g., "qazaq kiyimi" for Kazakh)
   - Create mixed-language combinations (native term + English descriptor)
   - Balance: approximately 40-50% native language, 40-50% English, 10-20% mixed

3. **Cultural Context**:
   - Include cultural identifiers (country name, ethnic group, etc.)
   - Use culturally appropriate terminology
   - Consider regional variations
   - Include both traditional and modern terms

4. **Variety and Coverage**:
   - Generate diverse keywords covering different aspects
   - Include synonyms and alternative phrasings
   - Consider different search intents (informational, visual, etc.)
   - Mix broad and specific terms

Generate 8-12 effective keywords per subcategory. The keywords MUST include both native language(s) and English terms. Return only a list of keyword strings, not objects.

Return only valid JSON."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, self.SYSTEM_PROMPT)
    
    def generate_keywords(
        self,
        category_name: str,
        subcategory_name: str,
        subcategory_description: str,
        country_or_culture: str = ""
    ) -> List[str]:
        country_context = f"\n\nCountry/Culture: {country_or_culture}" if country_or_culture else ""
        
        user_message = f"""Generate effective search keywords for internet search engines (Google, Bing, etc.) for the following subcategory:

Category: {category_name}
Subcategory: {subcategory_name}
Subcategory Description: {subcategory_description}{country_context}

REQUIREMENT: Generate 8-12 search keywords that MUST include BOTH native language(s) AND English keywords.

1. **Native Language Keywords** (40-50% of keywords):
   - Keywords in the native language(s) of {country_or_culture or "the country/culture"}
   - Example for Kazakhstan: Kazakh language using Cyrillic script (e.g., "Қазақстан дәстүрлі", "қазақ киімі", "дәстүрлі салт-дәстүр")
   - Use the native script appropriate for the country (Cyrillic, Arabic, Latin, etc.)
   - Include transliterations when helpful (e.g., "qazaq kiyimi" for Kazakh)
   - If the country has multiple official languages, include keywords in all major languages
   - These are essential for finding authentic, local cultural content

2. **English Keywords** (40-50% of keywords):
   - English translations and descriptions
   - International search terms
   - Include country/culture name when appropriate (e.g., "Kazakhstan traditional ceremony")
   - Use English for broader search coverage

3. **Mixed Language Keywords** (10-20% of keywords):
   - Combinations of native language + English (e.g., "Қазақстан traditional", "qazaq wedding ceremony")
   - Native term + English descriptor

4. **Search Optimization**:
   - Use specific, descriptive phrases (2-5 words)
   - Cover different aspects and variations of the subcategory
   - Think about what people actually search for in both languages
   - Optimize for Google Images and web search
   - Consider regional search patterns

IMPORTANT: The keyword list should be balanced between native language(s) and English. Both are essential for comprehensive data collection.

Return a simple list of keyword strings."""
        
        try:
            result = self.invoke(user_message)
            keywords = result.get("keywords", [])
            
            if keywords and isinstance(keywords[0], dict):
                keywords = [kw.get("keyword", "") for kw in keywords if kw.get("keyword")]
            
            keywords = [kw for kw in keywords if kw and isinstance(kw, str)]
            
            logger.info("Generated %d keywords for subcategory '%s'", len(keywords), subcategory_name)
            return keywords
        except Exception as e:
            logger.error("Error generating keywords: %s", e)
            return []

