#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import shutil
import sys
import time
import json
import logging
import requests
import argparse
import concurrent.futures

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GoogleImageScraper:
    """
    Scrapes images from Google Images for a given search keyword,
    using Selenium to automate Chrome. Downloads images to a local folder
    and saves metadata to JSON.
    """
    def __init__(
            self,
            webdriver_path,
            image_path,
            search_key="cat",
            number_of_images=1,
            headless=True,
            min_resolution=(0, 0),
            max_resolution=(1920, 1080),
            max_missed=10
    ):
        image_path = os.path.join(image_path, search_key)

        if not isinstance(number_of_images, int):
            logger.error("Number of images must be an integer.")
            return

        if not os.path.exists(image_path):
            logger.info("Image path not found. Creating a new folder.")
            os.makedirs(image_path)

        try:
            options = Options()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_experimental_option("excludeSwitches", ["enable-logging"])
            if headless:
                options.add_argument('--headless')

            service = Service(executable_path=webdriver_path)
            driver = webdriver.Chrome(service=service, options=options)

            driver.set_window_size(1400, 1050)
            driver.get("https://www.google.com")

            try:
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "W0wltc"))
                ).click()
            except Exception:
                pass

        except Exception as e:
            logger.error("Error launching web-driver: %s", e)
            exit("[ERR] It seems, your chromedriver version doesn't fit your Google Chrome. "
                 "Install correct: https://chromedriver.chromium.org/downloads")

        self.driver = driver
        self.search_key = search_key
        self.number_of_images = number_of_images
        self.webdriver_path = webdriver_path
        self.image_path = image_path
        self.url = "https://www.google.com/search?q=%s&source=lnms&tbm=isch" % search_key
        self.headless = headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_missed = max_missed
        self.metadata = []

    def find_image_urls(self):
        """
        Navigate to the Google Images page for self.search_key,
        collect image URLs, and store partial metadata.
        """
        logger.info("Searching images for: %s", self.search_key)
        self.driver.get(self.url)

        WebDriverWait(self.driver, 10).until(
            lambda d: len(d.find_elements(By.TAG_NAME, "img")) > 10
        )

        raw_thumbnails = self.driver.find_elements(By.TAG_NAME, "img")
        logger.debug("Number of thumbnails found: %d", len(raw_thumbnails))

        thumbnails = []
        for idx, thumb in enumerate(raw_thumbnails):
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
        self.driver.quit()
        return list(image_urls)

    def save_images(self, image_urls):
        """
        Download and save images from collected URLs. Also stores metadata info
        to a JSON file named metadata.json in the same directory.
        """
        logger.info("Saving %d images...", len(image_urls))

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
                            continue

                        try:
                            img.save(path)
                        except OSError:
                            img.convert("RGB").save(path)

                        logger.info("Saved: %s", path)

                        # Update metadata
                        for m in self.metadata:
                            if m["url"] == image_url:
                                m["filename"] = filename
                                m["filepath"] = path
                                break

                else:
                    logger.error("Wrong status-code: %s", response.status_code)

            except Exception as e:
                logger.error("Error saving image %d: %s", indx + 1, e)

        metadata_path = os.path.join(self.image_path, "metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=4)
            logger.info("Metadata saved to: %s", metadata_path)
        except Exception as e:
            logger.error("Error writing metadata.json: %s", e)

        logger.info("Download completed.")



def worker_thread(search_key, webdriver_path, image_path,
                  number_of_images, headless, min_resolution,
                  max_resolution, max_missed):
    scraper = GoogleImageScraper(
        webdriver_path=webdriver_path,
        image_path=image_path,
        search_key=search_key,
        number_of_images=number_of_images,
        headless=headless,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        max_missed=max_missed
    )
    image_urls = scraper.find_image_urls()
    logger.debug("[DEBUG] Found %d image URLs for %s", len(image_urls), search_key)
    scraper.save_images(image_urls)
    del scraper


def main():
    """CLI for GoogleImageScraper """
    parser = argparse.ArgumentParser(
        description="Scrape images from Google using Chrome via Selenium."
    )
    parser.add_argument("--search-keys", nargs="+", default=[],
                        help="Search keywords, space-separated. E.g.: --search-keys cat dog apple")
    parser.add_argument("--search-keys-file", default=None,
                        help="Path to JSON file containing search requests.")
    parser.add_argument("--images", type=int, default=30,
                        help="Number of images to scrape per keyword.")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode.")
    parser.add_argument("--min-resolution", type=int, nargs=2, default=[400, 400],
                        help="Minimum image resolution (width height).")
    parser.add_argument("--max-resolution", type=int, nargs=2, default=[8000, 8000],
                        help="Maximum image resolution (width height).")
    parser.add_argument("--max-missed", type=int, default=200,
                        help="Max number of times it can fail before stopping.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel threads to use.")
    parser.add_argument("--webdriver-dir", default="webdriver",
                        help="Directory where the chromedriver is stored.")
    parser.add_argument("--photos-dir", default="photos",
                        help="Directory to save the scraped images.")

    args = parser.parse_args()

    if args.webdriver_dir:
        potential_path = os.path.join(args.webdriver_dir, "chromedriver.exe" if os.name == "nt" else "chromedriver")
        if os.path.isfile(potential_path):
            webdriver_path = os.path.abspath(potential_path)
        else:
            wd_exec = shutil.which("chromedriver")
            if wd_exec is None:
                logger.error("chromedriver not found in PATH and not in %s" % args.webdriver_dir)
                sys.exit(1)
            webdriver_path = wd_exec
    else:
        wd_exec = shutil.which("chromedriver")
        if wd_exec is None:
            logger.error("chromedriver not found in PATH")
            sys.exit(1)
        webdriver_path = wd_exec

    image_path = os.path.normpath(os.path.join(os.getcwd(), args.photos_dir))
    search_keys = args.search_keys[:]

    if args.search_keys_file:
        try:
            with open(args.search_keys_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                if "keywords" in entry and isinstance(entry["keywords"], list):
                    search_keys.extend(entry["keywords"])
        except Exception as e:
            logger.error("Error reading search-keys-file: %s", e)
            sys.exit(1)

    search_keys = list(set(search_keys))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for sk in search_keys:
            executor.submit(
                worker_thread,
                sk,
                webdriver_path,
                image_path,
                args.images,
                args.headless,
                tuple(args.min_resolution),
                tuple(args.max_resolution),
                args.max_missed
            )

if __name__ == "__main__":
    main()
