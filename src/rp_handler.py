''' RunPod Serverless Handler '''
import os
import logging
import subprocess
import runpod
from runpod.serverless.utils import rp_download, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schema import INPUT_SCHEMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class JobHandler:
    def __init__(self):
        self.timeout = 300  # 5 minutes

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
            f"--garment={validated_input['garment']}",
            f"--mask={validated_input['mask']}",
            f"--model_img={validated_input['model_img']}",
            f"--output={validated_input['output']}",
            f"--width={validated_input.get('width', 1224)}",
            f"--height={validated_input.get('height', 1632)}",
            f"--num_inference_steps={validated_input.get('num_inference_steps', 50)}",
            f"--guidance_scale={validated_input['guidance_scale']}",
            f"--scheduler={validated_input.get('scheduler', 'K-LMS')}",
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

            # Download assets
            assets = self.download_assets(job['id'], validated)
            if not assets:
                return {"error": "Failed to download input files"}

            # Run prediction
            output_path = self.run_prediction(validated)
            if not output_path or not os.path.exists(output_path):
                return {"error": "Prediction failed"}

            # Return result
            return {"output": output_path}

        except Exception as e:
            logger.error(f"Job failed: {str(e)}")
            return {"error": str(e)}
        finally:
            rp_cleanup.clean(['input_objects'])

if __name__ == "__main__":
    handler = JobHandler()
    runpod.serverless.start({"handler": handler.handle_job})