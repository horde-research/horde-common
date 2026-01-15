import logging
from typing import Any, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, llm: BaseChatModel, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.parser = JsonOutputParser()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{user_message}")
        ])
    
    def invoke(self, user_message: str, **kwargs) -> Dict[str, Any]:
        try:
            prompt = self._create_prompt()
            chain = prompt | self.llm | self.parser
            result = chain.invoke({"user_message": user_message}, **kwargs)
            return result
        except Exception as e:
            logger.error("Error invoking agent: %s", e)
            raise

