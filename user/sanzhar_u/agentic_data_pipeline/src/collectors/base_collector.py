from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseCollector(ABC):
    @abstractmethod
    def collect(self, keyword: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        pass

