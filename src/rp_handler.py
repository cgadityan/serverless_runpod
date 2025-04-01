''' infer.py for runpod worker '''

import os
import base64
import predict
import argparse

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
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        logger.info("Uploading to S3 bucket")
        return upload_file_to_bucket(file_name, img_path)

    logger.info("Converting to base64")
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    
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
        validated_input['mask'], validated_input['garment'], validated_input['model_img'] = rp_download.download_files_from_urls(
            job['id'],
            [validated_input.get('mask', None), 
             validated_input.get('garment', None), 
             validated_input.get('model_img', None)]
        )  # pylint: disable=unbalanced-tuple-unpacking

        MODEL.NSFW = validated_input.get('nsfw', True)
        logger.info(f"NSFW setting: {MODEL.NSFW}")

        if validated_input['seed'] is None:
            validated_input['seed'] = int.from_bytes(os.urandom(2), "big")
            logger.info(f"Generated random seed: {validated_input['seed']}")

        logger.info("Starting model prediction")
        img_paths = MODEL.predict(
            prompt=validated_input["prompt"],
            garment = validated_input["garment"],
            mask=validated_input['mask'],
            model_img=validated_input["model_img"],
            output=validated_input["output"],
            width=validated_input.get('width', 1224),
            height=validated_input.get('height', 1632),    
            prompt_strength=validated_input['prompt_strength'],
            num_outputs=validated_input.get('num_outputs', 1),
            num_inference_steps=validated_input.get('num_inference_steps', 50),
            guidance_scale=validated_input['guidance_scale'],
            scheduler=validated_input.get('scheduler', "K-LMS"),
            seed=validated_input['seed']
        )

        logger.info(f"Image paths: {img_paths}")

        # Ensure img_paths is a list
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        elif not isinstance(img_paths, list):
            img_paths = list(img_paths)

        logger.info("Processing output images")
        job_output = []
        for index, img_path in enumerate(img_paths):
            file_name = f"{job['id']}_{index}.png"
            logger.info(f"Processing output image {index+1}/{len(img_paths)}")
            image_return = upload_or_base64_encode(file_name, img_path)

            job_output.append({
                "image": image_return,
                "seed": validated_input['seed'] + index
            })

        # Remove downloaded input objects
        logger.info("Cleaning up input objects")
        rp_cleanup.clean(['input_objects'])

        logger.info("Job completed successfully")
        return job_output

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# Grab args
parser = argparse.ArgumentParser()
parser.add_argument('--model_tag', type=str, default="black-forest-labs/FLUX.1-Fill-dev")

if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(f"Initializing with model tag: {args.model_tag}")

    MODEL = predict.Predictor(model_tag=args.model_tag)
    logger.info("Setting up model")
    MODEL.setup()

    logger.info("Starting runpod server")
    runpod.serverless.start({"handler": run})
