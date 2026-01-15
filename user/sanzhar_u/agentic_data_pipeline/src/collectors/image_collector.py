import os
import logging
from typing import Dict, Any

from .base_collector import BaseCollector

from .google_image_scraper import GoogleImageScraper

logger = logging.getLogger(__name__)


class ImageCollector(BaseCollector):
    
    def collect(self, keyword: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            os.makedirs(output_path, exist_ok=True)
            
            scraper = GoogleImageScraper(
                webdriver_path=config["webdriver_path"],
                image_path=output_path,
                search_key=keyword,
                number_of_images=config.get("number_of_images", 30),
                headless=config.get("headless", True),
                min_resolution=config.get("min_resolution", (400, 400)),
                max_resolution=config.get("max_resolution", (8000, 8000)),
                max_missed=config.get("max_missed", 200)
            )
            
            result = scraper.scrape()
            
            if result.get("success"):
                return {
                    "success": True,
                    "keyword": keyword,
                    "items_collected": result.get("saved", 0),
                    "output_path": os.path.join(output_path, keyword),
                    "urls_found": result.get("urls_found", 0),
                    "skipped": result.get("skipped", 0)
                }
            else:
                return {
                    "success": False,
                    "keyword": keyword,
                    "error": result.get("error", "Unknown error"),
                    "items_collected": 0,
                    "output_path": output_path
                }
        
        except Exception as e:
            logger.error("Error collecting images for keyword '%s': %s", keyword, e)
            return {
                "success": False,
                "keyword": keyword,
                "error": str(e),
                "items_collected": 0,
                "output_path": output_path
            }

