import os
import io
import time
import json
import logging
import requests
from typing import List, Tuple, Optional

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class GoogleImageScraper:
    def __init__(
        self,
        webdriver_path: str,
        image_path: str,
        search_key: str = "cat",
        number_of_images: int = 1,
        headless: bool = True,
        min_resolution: Tuple[int, int] = (0, 0),
        max_resolution: Tuple[int, int] = (1920, 1080),
        max_missed: int = 10
    ):
        self.search_key = search_key
        self.number_of_images = number_of_images
        self.webdriver_path = webdriver_path
        self.image_path = os.path.join(image_path, search_key)
        self.headless = headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_missed = max_missed
        self.metadata = []
        self.driver: Optional[webdriver.Chrome] = None
        
        if not isinstance(number_of_images, int):
            raise ValueError("Number of images must be an integer.")
        
        if not os.path.exists(self.image_path):
            logger.info("Image path not found. Creating a new folder.")
            os.makedirs(self.image_path, exist_ok=True)
    
    def _initialize_driver(self) -> webdriver.Chrome:
        try:
            options = Options()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_experimental_option("excludeSwitches", ["enable-logging"])
            if self.headless:
                options.add_argument('--headless')
            
            service = Service(executable_path=self.webdriver_path)
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_window_size(1400, 1050)
            driver.get("https://www.google.com")
            
            try:
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "W0wltc"))
                ).click()
            except Exception:
                pass
            
            return driver
        except Exception as e:
            logger.error("Error launching web-driver: %s", e)
            raise RuntimeError(
                "It seems your chromedriver version doesn't fit your Google Chrome. "
                "Install correct: https://chromedriver.chromium.org/downloads"
            ) from e
    
    def find_image_urls(self) -> List[str]:
        if self.driver is None:
            self.driver = self._initialize_driver()
        
        logger.info("Searching images for: %s", self.search_key)
        url = f"https://www.google.com/search?q={self.search_key}&source=lnms&tbm=isch"
        self.driver.get(url)
        
        WebDriverWait(self.driver, 10).until(
            lambda d: len(d.find_elements(By.TAG_NAME, "img")) > 10
        )
        
        raw_thumbnails = self.driver.find_elements(By.TAG_NAME, "img")
        logger.debug("Number of thumbnails found: %d", len(raw_thumbnails))
        
        thumbnails = []
        for thumb in raw_thumbnails:
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", thumb)
                time.sleep(0.1)
                width = self.driver.execute_script(
                    "return arguments[0].naturalWidth;",
                    thumb
                )
                src = thumb.get_attribute("src")
                if src and width and width >= 80:
                    thumbnails.append(thumb)
            except Exception:
                continue
        
        logger.info("Number of valid thumbnails: %d", len(thumbnails))
        
        image_urls = set()
        missed_count = 0
        
        for index, thumbnail in enumerate(thumbnails):
            if len(image_urls) >= self.number_of_images:
                break
            
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", thumbnail)
                time.sleep(0.1)
                try:
                    thumbnail.click()
                    logger.debug("Click on preview #%d", index + 1)
                except Exception:
                    missed_count += 1
                    if missed_count > self.max_missed:
                        logger.critical("Too many clicks without positive result.")
                        break
                    continue
                
                WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img.iPVvYb"))
                )
                images = self.driver.find_elements(By.CSS_SELECTOR, "img.iPVvYb")
                
                for img in images:
                    full_src = img.get_attribute("src")
                    natural_width = self.driver.execute_script(
                        "return arguments[0].naturalWidth",
                        img
                    )
                    if full_src and full_src.startswith("http") and natural_width > 300:
                        if full_src not in image_urls:
                            image_urls.add(full_src)
                            outer_html = img.get_attribute("outerHTML")
                            self.metadata.append({
                                "id": len(self.metadata),
                                "url": full_src,
                                "html": outer_html
                            })
                            logger.info("[+] Saving: %s", full_src)
                        break
            
            except Exception as e:
                logger.error("A thumbnail error #%d: %s", index + 1, e)
                continue
        
        logger.info("Total URLs gathered: %d", len(image_urls))
        return list(image_urls)
    
    def save_images(self, image_urls: List[str]) -> dict:
        logger.info("Saving %d images...", len(image_urls))
        
        saved_count = 0
        skipped_count = 0
        
        for indx, image_url in enumerate(image_urls):
            try:
                response = requests.get(image_url, timeout=5)
                if response.status_code == 200:
                    with Image.open(io.BytesIO(response.content)) as img:
                        filename = f"{self.search_key}{indx}.{img.format.lower()}"
                        path = os.path.join(self.image_path, filename)
                        
                        if (img.size[0] < self.min_resolution[0] or
                                img.size[1] < self.min_resolution[1] or
                                img.size[0] > self.max_resolution[0] or
                                img.size[1] > self.max_resolution[1]):
                            logger.warning(
                                "Image %s does not fit size constraints %s -> SKIPPED...",
                                filename,
                                img.size
                            )
                            skipped_count += 1
                            continue
                        
                        try:
                            img.save(path)
                        except OSError:
                            img.convert("RGB").save(path)
                        
                        logger.info("Saved: %s", path)
                        saved_count += 1
                        
                        # Update metadata
                        for m in self.metadata:
                            if m["url"] == image_url:
                                m["filename"] = filename
                                m["filepath"] = path
                                break
                else:
                    logger.error("Wrong status-code: %s", response.status_code)
                    skipped_count += 1
            
            except Exception as e:
                logger.error("Error saving image %d: %s", indx + 1, e)
                skipped_count += 1
        
        metadata_path = os.path.join(self.image_path, "metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=4)
            logger.info("Metadata saved to: %s", metadata_path)
        except Exception as e:
            logger.error("Error writing metadata.json: %s", e)
        
        logger.info("Download completed.")
        return {
            "saved": saved_count,
            "skipped": skipped_count,
            "total": len(image_urls)
        }
    
    def scrape(self) -> dict:
        try:
            image_urls = self.find_image_urls()
            stats = self.save_images(image_urls)
            return {
                "success": True,
                "keyword": self.search_key,
                "urls_found": len(image_urls),
                **stats
            }
        except Exception as e:
            logger.error("Error during scraping: %s", e)
            return {
                "success": False,
                "keyword": self.search_key,
                "error": str(e)
            }
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

