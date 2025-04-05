<<<<<<< HEAD
<div align="center">

<h1>Flux Fill Dev 1 model | Worker</h1>

[![Docker Image](https://github.com/runpod-workers/worker-stable_diffusion_v1/actions/workflows/CD-docker_release.yml/badge.svg)](https://github.com/runpod-workers/worker-stable_diffusion_v1/actions/workflows/CD-docker_release.yml)

</div>

## RunPod Endpoint

This repository contains the worker for the VTON-FLUX Endpoints. The following docs can be referenced to make direct calls to the running endpoints on runpod.io


## Docker Image

The docker image requires two build arguments `MODEL_URL` and `Model_TAG` to build the image. The `MODEL_URL` is the url of the model repository and the `Model_TAG` is the tag of the model repository.

```bash
docker build --build-arg MODEL_URL=https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev --build-arg MODEL_TAG=black-forest-labs/FLUX.1-Fill-dev -t runwayml/FLUX.1-Fill-dev .
```

## Continuous Deployment

This worker follows a modified version of the [worker template](https://github.com/runpod-workers/worker-template) where the Docker build workflow contains additional models to be built and pushed.

=======
# serverless_runpod
>>>>>>> 5204476ce71e8f31439f4d7f1fa8de65b860bc3d
