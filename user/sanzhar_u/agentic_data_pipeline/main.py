import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core import PipelineBuilder
from src.config import create_image_collection_config
from src.nodes import (
    extract_categories_node,
    generate_subcategories_node,
    generate_keywords_node,
    collect_data_node
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_llm(provider: str = None, model: str = None, **kwargs):
    provider = provider or kwargs.get("provider") or os.getenv("LLM_PROVIDER", "google")
    
    if provider.lower() == "openai":
        model = model or kwargs.get("model") or os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = kwargs.get("api_key") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0.7)
    
    elif provider.lower() == "anthropic":
        model = model or kwargs.get("model") or os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
        api_key = kwargs.get("api_key") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0.7)
    
    elif provider.lower() == "google":
        model = model or kwargs.get("model") or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        api_key = kwargs.get("api_key") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable not set")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.7)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'google'")


def find_chromedriver(webdriver_dir: str = None) -> str:
    if webdriver_dir:
        potential_path = os.path.join(
            webdriver_dir,
            "chromedriver.exe" if os.name == "nt" else "chromedriver"
        )
        if os.path.isfile(potential_path):
            return os.path.abspath(potential_path)
    
    wd_exec = shutil.which("chromedriver")
    if wd_exec is None:
        raise FileNotFoundError(
            "chromedriver not found. Please install it or specify --webdriver-dir"
        )
    return wd_exec


def save_results(results: dict, output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to: %s", output_path)
    except Exception as e:
        logger.error("Error saving results: %s", e)


def create_graph(
    llm_provider: str = "google",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    webdriver_path: Optional[str] = None,
    data_path: str = "./data",
    items_per_keyword: int = 30,
    headless: bool = True,
    min_resolution: tuple = (400, 400),
    max_resolution: tuple = (8000, 8000),
    max_missed: int = 200
):
    llm = get_llm(provider=llm_provider, model=llm_model, api_key=llm_api_key)
    
    if webdriver_path is None:
        try:
            webdriver_path = find_chromedriver()
        except FileNotFoundError:
            webdriver_path = "/usr/local/bin/chromedriver"
    
    config = create_image_collection_config(
        llm=llm,
        webdriver_path=webdriver_path,
        data_path=os.path.abspath(data_path),
        number_of_images=items_per_keyword,
        headless=headless,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        max_missed=max_missed
    )
    
    builder = PipelineBuilder(config.to_dict())
    
    builder.set_entry_point("extract_categories")
    builder.add_node("extract_categories", extract_categories_node)
    builder.add_node("generate_subcategories", generate_subcategories_node)
    builder.add_node("generate_keywords", generate_keywords_node)
    builder.add_node("collect_data", collect_data_node)
    builder.connect_to_end("collect_data")
    
    workflow = builder.build()
    
    return workflow.graph, builder, config


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <text>")
        logger.error("   or: python main.py --file <path-to-text-file>")
        sys.exit(1)
    
    if sys.argv[1] == "--file" or sys.argv[1] == "-f":
        if len(sys.argv) < 3:
            logger.error("Error: --file requires a file path")
            sys.exit(1)
        try:
            with open(sys.argv[2], "r", encoding="utf-8") as f:
                free_text = f.read().strip()
        except Exception as e:
            logger.error("Error reading text file: %s", e)
            sys.exit(1)
    else:
        free_text = sys.argv[1]
    
    if not free_text:
        logger.error("No text provided")
        sys.exit(1)
    
    llm_provider = os.getenv("LLM_PROVIDER", "google")
    llm_model = os.getenv("LLM_MODEL")
    llm_api_key = os.getenv("LLM_API_KEY")
    
    items_per_keyword = int(os.getenv("ITEMS_PER_KEYWORD", "30"))
    headless = os.getenv("HEADLESS", "true").lower() in ("true", "1", "yes")
    
    min_res_env = os.getenv("MIN_RESOLUTION", "400,400").split(",")
    min_resolution = tuple(int(x.strip()) for x in min_res_env[:2])
    
    max_res_env = os.getenv("MAX_RESOLUTION", "8000,8000").split(",")
    max_resolution = tuple(int(x.strip()) for x in max_res_env[:2])
    
    max_missed = int(os.getenv("MAX_MISSED", "200"))
    data_dir = os.getenv("DATA_DIR", "data")
    output_file = os.getenv("OUTPUT_FILE", "results.json")
    webdriver_dir = os.getenv("WEBDRIVER_DIR")
    
    try:
        logger.info("Initializing LLM (provider: %s)...", llm_provider)
        llm = get_llm(provider=llm_provider, model=llm_model, api_key=llm_api_key)
    except Exception as e:
        logger.error("Error initializing LLM: %s", e)
        sys.exit(1)
    
    try:
        webdriver_path = find_chromedriver(webdriver_dir)
        logger.info("Using chromedriver: %s", webdriver_path)
    except Exception as e:
        logger.error("Error finding chromedriver: %s", e)
        sys.exit(1)
    
    logger.info("Building pipeline...")
    graph, builder, config = create_graph(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        webdriver_path=webdriver_path,
        data_path=data_dir,
        items_per_keyword=items_per_keyword,
        headless=headless,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        max_missed=max_missed
    )
    
    initial_state = builder.create_initial_state(free_text=free_text)
    
    logger.info("Starting workflow with text: %s", free_text[:100] + "..." if len(free_text) > 100 else free_text)
    try:
        results = graph.invoke(initial_state)
        
        save_results(results, output_file)
        
        logger.info("\n" + "="*60)
        logger.info("WORKFLOW SUMMARY")
        logger.info("="*60)
        logger.info("Categories extracted: %d", len(results.get("categories", [])))
        
        total_subcategories = sum(
            len(subs) for subs in results.get("category_subcategories", {}).values()
        )
        logger.info("Subcategories generated: %d", total_subcategories)
        
        total_keywords = sum(
            len(keywords)
            for subcats in results.get("category_subcategory_keywords", {}).values()
            for keywords in subcats.values()
        )
        logger.info("Keywords generated: %d", total_keywords)
        
        collection_results = results.get("collection_results", [])
        successful = sum(1 for r in collection_results if r.get("success"))
        logger.info("Successful collections: %d / %d", successful, len(collection_results))
        
        total_items = sum(r.get("items_collected", 0) for r in collection_results)
        logger.info("Total items collected: %d", total_items)
        logger.info("="*60)
        
    except Exception as e:
        logger.error("Error running workflow: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

try:
    graph, _, _ = create_graph(
        llm_provider=os.getenv("LLM_PROVIDER", "google"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        webdriver_path=None,
        data_path=os.getenv("DATA_DIR", "./data"),
        items_per_keyword=int(os.getenv("ITEMS_PER_KEYWORD", "30")),
        headless=os.getenv("HEADLESS", "true").lower() == "true",
        min_resolution=(400, 400),
        max_resolution=(8000, 8000),
        max_missed=200
    )
except Exception as e:
    logger.warning(f"Could not create graph for LangGraph Studio: {e}")
    graph = None
