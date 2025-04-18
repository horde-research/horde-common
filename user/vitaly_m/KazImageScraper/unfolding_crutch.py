import os
import shutil
import logging
from typing import List

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_env_vars() -> tuple[str, str, str]:
    """Download environmental variables from .env."""
    load_dotenv()
    csv_path = os.getenv("CSV_PATH")
    src_root = os.getenv("SRC_ROOT")
    dst_root = os.getenv("DST_ROOT")

    if not all([csv_path, src_root, dst_root]):
        logger.critical("Error reading a .env. Check the .env.")
        raise EnvironmentError("Missing required environment variables.")

    return csv_path, src_root, dst_root


def clean_keywords(keywords_str: str) -> List[str]:
    """Reading the keywords."""
    cleaned = str(keywords_str).replace('\n', ' ')
    return [kw.strip().strip('"').strip("'") for kw in cleaned.split(',') if kw.strip()]


def move_folders_by_keywords(csv_path: str, src_root: str, dst_root: str) -> None:
    """Reading CSV and unfolding folders accordingly keywords."""
    dataframe = pd.read_csv(csv_path)

    for _, row in dataframe.iterrows():
        work_cat = row['work_cat_name'].strip()
        work_subcat = row['work_subcat_name'].strip()
        keywords = clean_keywords(row['keywords'])

        target_dir = os.path.join(dst_root, work_cat, work_subcat)
        os.makedirs(target_dir, exist_ok=True)

        for keyword in keywords:
            src_path = os.path.join(src_root, keyword)
            dst_path = os.path.join(target_dir, keyword)

            if not os.path.isdir(src_path):
                logger.warning("[MISS] Not founded: %s", src_path)
                continue

            if os.path.exists(dst_path):
                logger.info("[SKIP] Already exists: %s", dst_path)
                continue

            logger.info("[TRY] Unfolding: %s → %s", src_path, dst_path)
            try:
                shutil.move(src_path, dst_path)
                logger.info("[OK ] Successfully unfold.")
            except OSError as exc:
                logger.error("[ERR] Error during unfolding: %s", exc)


def main():
    csv_path, src_root, dst_root = load_env_vars()
    move_folders_by_keywords(csv_path, src_root, dst_root)

if __name__ == "__main__":
    main()