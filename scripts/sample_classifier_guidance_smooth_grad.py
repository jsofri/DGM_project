import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.dirname(script_dir))

# modify environment before importing non-builtin module
os.environ["REPO_ROOT"] = os.environ.get("REPO_ROOT", repo_root)

# add to python path
sys.path.append(f"{repo_root}/Diffusion_models_from_scratch")
from src.infer import infer


if __name__ == "__main__":
    classifier_guidance = "openai"
    if len(sys.argv) > 1:
        classifier_guidance = sys.argv[1]
        if classifier_guidance not in ("openai", "resnet18"):
            print(f"Usage: {sys.argv[0]} [openai|resnet18]")


    # manipulate sys.argv to make the module work as expected
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
        f"--classifier_guidance",
        classifier_guidance,
        "--guidance",
        "4",
        "--smooth_grad",
        "1"
    ]

    infer()

    print()
    print("Check the following files:")
    print(f"sample diffusion GIF: {repo_root}/diffusion.gif")
    print(f"sample: {repo_root}/fig.png")
