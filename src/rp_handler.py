''' RunPod Serverless Handler '''
import os
import logging
import subprocess
import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from rp_schema import INPUT_SCHEMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

class JobHandler:
    def __init__(self):
        self.timeout = 150  # 2.5 minutes

    def validate_input(self, job_input):
        ''' Validate input against schema '''
        validation = validate(job_input, INPUT_SCHEMA)
        if 'errors' in validation:
            logger.error(f"Validation failed: {validation['errors']}")
            return None
        return validation['validated_input']

    def download_assets(self, job_id, inputs):
        ''' Download input files '''
        try:
            return rp_download.download_files_from_urls(
                job_id,
                [inputs['mask'], inputs['garment'], inputs['model_img']]
            )
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return None

    def run_prediction(self, validated_input):
        ''' Execute distributed prediction '''
        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--nnodes=1",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "predict.py",
            "--ulysses_degree=2",
            "--ring_degree=1",
            f"--garment={validated_input['garment']}",
            f"--mask={validated_input['mask']}",
            f"--model_img={validated_input['model_img']}",
            f"--output={validated_input['output']}",
            f"--width={validated_input.get('width', 1224)}",
            f"--height={validated_input.get('height', 1632)}",
            f"--num_inference_steps={validated_input.get('num_inference_steps', 50)}",
            f"--guidance_scale={validated_input['guidance_scale']}",
            f"--scheduler={validated_input.get('scheduler', 'FMEULER-D')}",
            f"--seed={validated_input.get('seed', 42)}",
            f"--prompt={validated_input['prompt']}"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True
            )
            
            # Parse output
            for line in result.stdout.split('\n'):
                if line.startswith("PREDICTION_SUCCESS|"):
                    return line.split("|")[1].strip()
            
            logger.error("No valid output detected")
            return None

        except subprocess.TimeoutExpired:
            logger.error("Prediction timed out")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Prediction failed: {e.stderr}")
            return None

    def handle_job(self, job):
        ''' Main job handler '''
        try:
            # Validate input
            validated = self.validate_input(job['input'])
            if not validated:
                return {"error": "Invalid input parameters"}

            # Download assets and get LOCAL PATHS
            downloaded_paths = self.download_assets(job['id'], validated)
            if not downloaded_paths:
                return {"error": "Failed to download input files"}

            # Update validated input with LOCAL PATHS
            validated['mask'], validated['garment'], validated['model_img'] = downloaded_paths
            # logger.info("Downloaded items paths: ",downloaded_paths)

            # Run prediction
            output_paths = self.run_prediction(validated)
            # if not output_paths or not os.path.exists(output_paths[]):
            #     return {"error": "Prediction failed"}

            # Return result
            # return {"output": output_path}

            # Ensure output_paths is a list
            if isinstance(output_paths, str):
                output_paths = [output_paths]
            elif not isinstance(output_paths, list):
                output_paths = list(output_paths)

            logger.info("Processing output images")
            job_output = []
            for index, img_path in enumerate(output_paths):
                if not os.path.exists(img_path):
                    logger.error(f"Output image not found: {img_path}")
                    continue
                    
                file_name = f"{job['id']}_{index}.png"
                logger.info(f"Processing output image {index+1}/{len(output_paths)}")
                try:
                    image_return = upload_or_base64_encode(file_name, img_path)
                    job_output.append({
                        "image": image_return,
                        "seed": validated['seed'] + index
                    })
                except Exception as e:
                    logger.error(f"Error processing output image {img_path}: {str(e)}")
            


        except Exception as e:
            logger.error(f"Job failed: {str(e)}")
            return {"error": str(e)}
        finally:
            rp_cleanup.clean(['input_objects'])

if __name__ == "__main__":
    handler = JobHandler()
    runpod.serverless.start({"handler": handler.handle_job})