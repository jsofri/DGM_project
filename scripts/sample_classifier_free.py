import torch
from diffusers import ConsistencyModelPipeline

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the cd_imagenet64_l2 checkpoint
model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
try:
    pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Ensure that the 'safetensors' library is correctly installed.")
    exit(1)
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
    print("There might be an issue with the model file encoding. Try re-downloading the model.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)

pipe.to(device)

# ImageNet-64 class label for golden retriever
class_id = 207  # Class ID for golden retriever

# Generate and save 10 images
for i in range(10):
    try:
        image = pipe(num_inference_steps=1, class_labels=class_id).images[0]
        image.save(f"golden_retriever_{i}.png")
    except Exception as e:
        print(f"An error occurred while generating or saving image {i}: {e}")

print("Generated 10 images of golden retriever.")