# Generative Models Course Project
This repo is based on 2 code bases for ImageNet 64x64 image generation diffusion based-models, with several modification as a part of an academic project about diffusion models
Project structure:
- guided-diffusion - Based on https://github.com/openai/guided-diffusion
- Diffusion_models_from_scratch - Based on https://github.com/gmongaras/Diffusion_models_from_scratch
- scripts - custom scripts for the project

## Setup
1. clone the repository
2. `python3 -m pip install -r requirements.txt`
2. download pretrained models and place in the corresponding folders
    - guided-diffusion/models:
        - classifier https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
        - diffusion https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
    - Diffusion_models_from_scratch/models:
        - `model_479e_600000s.pkl` and `model_params_479e_600000s.json` from https://drive.google.com/drive/folders/1NvM8S7U8uZ_TbSHE_p0wPlss2Mchwjab


## Usage

### sample using the original classifier-free guidance model
```
python3 ./scripts/sample_classifier_free.py
```

### sample using our "hybrid" model (unconditional classifier-free and classifier guidance)
```
python3 ./scripts/sample_classifier_guidance.py
```
