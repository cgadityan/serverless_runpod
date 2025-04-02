''' infer.py for runpod worker '''

import os
import base64
import predict
import argparse
import subprocess
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup

from rp_schema import INPUT_SCHEMA
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def upload_or_base64_encode(file_name, img_path):
    """
    Uploads image to S3 bucket if it is available, otherwise returns base64 encoded image.
    """
    logger.info(f"Processing file {file_name}")
    
    # Check if file exists
    if not os.path.exists(img_path):
        logger.error(f"File not found: {img_path}")
        raise FileNotFoundError(f"File not found: {img_path}")
        
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        logger.info("Uploading to S3 bucket")
        return upload_file_to_bucket(file_name, img_path)

    logger.info("Converting to base64")
    try:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding file {img_path}: {str(e)}")
        raise

def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    
    if not job or 'id' not in job or 'input' not in job:
        logger.error("Invalid job format")
        return {"error": "Invalid job format"}
        
    try:
        logger.info(f"Starting job {job['id']}")
        job_input = job['input']

        # Input validation
        logger.info("Validating input")
        validated_input = validate(job_input, INPUT_SCHEMA)

        if 'errors' in validated_input:
            logger.error(f"Input validation failed: {validated_input['errors']}")
            return {"error": validated_input['errors']}
        validated_input = validated_input['validated_input']

        # Download input objects
        logger.info("Downloading input files")
        try:
            validated_input['mask'], validated_input['garment'], validated_input['model_img'] = rp_download.download_files_from_urls(
                job['id'],
                [validated_input.get('mask', None), 
                 validated_input.get('garment', None), 
                 validated_input.get('model_img', None)]
            )  # pylint: disable=unbalanced-tuple-unpacking
        except Exception as e:
            logger.error(f"Error downloading input files: {str(e)}")
            return {"error": f"Failed to download input files: {str(e)}"}

        if validated_input['seed'] is None:
            validated_input['seed'] = int.from_bytes(os.urandom(2), "big")
            logger.info(f"Generated random seed: {validated_input['seed']}")


        logger.info("Starting model prediction with distributed processing")
        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "predict.py",
            f"--prompt={validated_input['prompt']}",
            f"--garment={validated_input['garment']}", 
            f"--mask={validated_input['mask']}",
            f"--model_img={validated_input['model_img']}",
            f"--output={validated_input['output']}",
            f"--width={validated_input.get('width', 1224)}",
            f"--height={validated_input.get('height', 1632)}",
            f"--model black-forest-labs/FLUX.1-Fill-dev ",
            f"--ulysses_degree 2",
            f"--ring_degree 1",
            f"--prompt_strength={validated_input['prompt_strength']}",
            f"--num_outputs={validated_input.get('num_outputs', 1)}",
            f"--num_inference_steps={validated_input.get('num_inference_steps', 50)}",
            f"--guidance_scale={validated_input['guidance_scale']}",
            f"--scheduler={validated_input.get('scheduler', 'K-LMS')}",
            f"--seed={validated_input['seed']}",
        ]
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Process stdout: {process.stdout}")
            logger.info(f"Process stderr: {process.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Prediction process failed: {e.stderr}")
            raise RuntimeError(f"Prediction process failed: {e.stderr}")
            
        # Parse output paths from stdout
        img_paths = process.stdout.strip().split('\n')
        if not img_paths or img_paths[0] == '':
            logger.error("No output paths received from prediction")
            raise RuntimeError("No output paths received from prediction")

        logger.info(f"Image paths: {img_paths}")

        # Ensure img_paths is a list
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        elif not isinstance(img_paths, list):
            img_paths = list(img_paths)

        logger.info("Processing output images")
        job_output = []
        for index, img_path in enumerate(img_paths):
            if not os.path.exists(img_path):
                logger.error(f"Output image not found: {img_path}")
                continue
                
            file_name = f"{job['id']}_{index}.png"
            logger.info(f"Processing output image {index+1}/{len(img_paths)}")
            try:
                image_return = upload_or_base64_encode(file_name, img_path)
                job_output.append({
                    "image": image_return,
                    "seed": validated_input['seed'] + index
                })
            except Exception as e:
                logger.error(f"Error processing output image {img_path}: {str(e)}")

        # Remove downloaded input objects
        logger.info("Cleaning up input objects")
        try:
            rp_cleanup.clean(['input_objects'])
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

        if not job_output:
            logger.error("No output images were successfully processed")
            return {"error": "No output images were successfully processed"}

        logger.info("Job completed successfully")
        return job_output

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# Grab args
parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', type=str, default="black-forest-labs/FLUX.1-Fill-dev")

if __name__ == "__main__":
    try:
        args = parser.parse_args()
        logger.info(f"Initializing with model tag: {args.model_tag}")

        # MODEL = predict.Predictor(model_tag=args.model_tag)
        # logger.info("Setting up model")
        # MODEL.setup()

        logger.info("Starting runpod server")
        runpod.serverless.start({"handler": run})
    except Exception as e:
        logger.critical(f"Fatal error during startup: {str(e)}")
        raise
