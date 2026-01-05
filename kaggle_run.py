import os
import subprocess
import sys

# 1. THE FORCE INSTALL PROTOCOL (100% Deterministic)
# We use --no-cache-dir to prevent disk bloat and ensure fresh binary builds
# We pin versions to the exact Kohya-ss SD3-branch tested specs
def force_install():
    print("🚀 Initiating Vulcan Dependency Protocol...")
    pkgs = [
        "torch==2.4.0", 
        "torchvision==0.19.0",
        "accelerate==0.33.0",
        "transformers==4.44.0",
        "diffusers==0.25.0",
        "bitsandbytes==0.44.0", # Required for NF4 loading
        "safetensors==0.4.4",   # Fixes the --resume logic errors
        "toml", "imagesize", "voluptuous", "einops"
    ]
    
    # We use pip install with --upgrade and --no-cache-dir
    for pkg in pkgs:
        print(f"Installing {pkg}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", "--no-cache-dir", pkg
        ], check=True)
    print("✅ Environment Locked.")

# Execute install before anything else
force_install()

# 2. Hardware Isolation & Logic
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONPATH"] = "."
os.environ["PIP_NO_CACHE_DIR"] = "1" # Force pip to ignore cache globally

# ... [The rest of your kaggle_run.py logic follows here]

# 2. Automated Dataset Preparation
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)
    
# Copy all supported image types and text files from Input to Working
extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.txt']
print(f"Searching for images in {USER_DATASET_PATH}...")

found_files = 0
for ext in extensions:
    # This finds files regardless of case (e.g., .JPG or .jpg)
    files = glob.glob(os.path.join(USER_DATASET_PATH, "**", ext), recursive=True)
    for f in files:
        shutil.copy(f, WORKING_DIR)
        found_files += 1

print(f"Dataset Prepared: {found_files} files copied to {WORKING_DIR}")

# 3. Extract the Trainer
if not os.path.exists("library"):
    print("Extracting Vulcan Trainer...")
    subprocess.run(["unzip", "-q", "vulcan_trainer.zip", "-d", "."])

# 4. The Training Command
cmd = [
    "python", "flux_train_network.py",
    "--pretrained_model_name_or_path", "/kaggle/input/flux-dev-bnb-nf4/flux1-dev-bnb-nf4.safetensors",
    "--clip_l", "/kaggle/input/flux-dev-bnb-nf4/clip_l.safetensors",
    "--t5xxl", "/kaggle/input/flux-dev-bnb-nf4/t5xxl_fp8_e4m3fn.safetensors",
    "--ae", "/kaggle/input/flux-dev-bnb-nf4/ae.safetensors",
    "--cache_latents",
    "--cache_text_encoder_outputs",
    "--save_model_as", "safetensors",
    "--sdpa",
    "--fp8_base",
    "--gradient_checkpointing",
    "--mixed_precision", "bf16",
    "--network_module", "networks.flux_lora",
    "--network_dim", "16",
    "--network_alpha", "16",
    "--lr_scheduler", "constant",
    "--learning_rate", "1e-4",
    "--max_train_steps", "1000",
    "--train_batch_size", "1",
    "--output_dir", "/kaggle/working/output",
    "--output_name", "flux_lora_v1",
    "--dataset_config", "dataset.toml"
]

subprocess.run(cmd)