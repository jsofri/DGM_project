import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.dirname(script_dir))

# modify environment before importing non-builtin module
os.environ["REPO_ROOT"] = repo_root

# add to python path
sys.path.append(f"{repo_root}/Diffusion_models_from_scratch")
from src.infer import infer



if __name__ == "__main__":


    # manipulate sys.argv to make the module work sa expected
    sys.argv = [
        f"{repo_root}/Diffusion_models_from_scratch/src/infer.py",
        f"--loadDir",
        f"{repo_root}/Diffusion_models_from_scratch/models",
        f"--loadFile",
        f"model_479e_600000s.pkl",
        f"--loadDefFile",
        f"model_params_479e_600000s.json",
        f"--class_label",
        f"207",  # Golden retriever, hopefully...
        f"--grads_file",
        f"grads",
        f"--classifier-guidance",
        f"True"
    ]

    infer()

    print()
    print("Check the following files:")
    print(f"sample diffusion GIF: {repo_root}/diffusion.gif")
    print(f"sample: {repo_root}/fig.png")
