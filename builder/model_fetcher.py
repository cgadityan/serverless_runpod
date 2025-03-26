'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''

import os
import shutil
import requests
import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse

from diffusers import FluxFillPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = "hf-cache"


def download_model(model_url: str):
    '''
    Downloads the model from the URL passed in.
    '''
    try:
        logger.info(f"Starting model download from: {model_url}")
        model_cache_path = Path(MODEL_CACHE_DIR)
        
        if model_cache_path.exists():
            logger.info(f"Removing existing cache directory: {model_cache_path}")
            shutil.rmtree(model_cache_path)
        
        logger.info(f"Creating cache directory: {model_cache_path}")
        model_cache_path.mkdir(parents=True, exist_ok=True)

        # Check if the URL is from huggingface.co
        parsed_url = urlparse(model_url)
        if parsed_url.netloc == "huggingface.co":
            model_id = f"{parsed_url.path.strip('/')}"
            logger.info(f"Detected Hugging Face model ID: {model_id}")
            
            logger.info(f"Downloading model from Hugging Face: {model_id}")
            FluxFillPipeline.from_pretrained(
                model_id,
                cache_dir=model_cache_path,
            )
            logger.info(f"Model successfully downloaded to {model_cache_path}")
        else:
            logger.info(f"Downloading model from direct URL: {model_url}")
            downloaded_model = requests.get(model_url, stream=True, timeout=600)
            downloaded_model.raise_for_status()  # Raise exception for HTTP errors
            
            model_file_path = model_cache_path / "model.zip"
            logger.info(f"Saving model to: {model_file_path}")
            
            with open(model_file_path, "wb") as f:
                total_size = 0
                for chunk in downloaded_model.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        logger.info(f"Downloaded: {total_size / (1024*1024):.2f} MB")
            
            logger.info(f"Model successfully downloaded to {model_file_path}")
        
        # Verify the download
        if os.path.exists(model_cache_path):
            logger.info(f"Verification successful: Model cache directory exists at {model_cache_path}")
            for item in os.listdir(model_cache_path):
                logger.info(f"Cache contains: {item}")
        else:
            logger.error(f"Verification failed: Model cache directory not found at {model_cache_path}")
            
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_url", type=str,
    default="https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev",
    help="URL of the model to download."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_url)
