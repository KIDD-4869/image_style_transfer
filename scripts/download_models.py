#!/usr/bin/env python3
"""
Model Download Script for Enhanced Ghibli Processor
Downloads required models from Hugging Face Hub
"""
import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️ huggingface_hub not installed. Install with: pip install huggingface_hub")


def download_model(repo_id: str, cache_dir: str, model_name: str):
    """Download a model from Hugging Face Hub"""
    print(f"\n📥 Downloading {model_name}...")
    print(f"   Repository: {repo_id}")
    print(f"   Cache directory: {cache_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True
        )
        print(f"✅ {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for Enhanced Ghibli Processor")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models/cache",
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["base", "controlnet", "all"],
        default=["all"],
        help="Which models to download"
    )
    
    args = parser.parse_args()
    
    if not HF_HUB_AVAILABLE:
        print("❌ Cannot download models without huggingface_hub")
        print("   Install with: pip install huggingface_hub")
        sys.exit(1)
    
    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Enhanced Ghibli Processor - Model Download")
    print("=" * 60)
    
    models_to_download = []
    
    if "all" in args.models or "base" in args.models:
        models_to_download.append({
            "repo_id": "Linaqruf/anything-v3.0",
            "name": "Anything V3.0 (Base Anime Model)"
        })
    
    if "all" in args.models or "controlnet" in args.models:
        models_to_download.append({
            "repo_id": "lllyasviel/control_v11p_sd15_canny",
            "name": "ControlNet Canny"
        })
        models_to_download.append({
            "repo_id": "lllyasviel/control_v11f1p_sd15_depth",
            "name": "ControlNet Depth"
        })
    
    print(f"\n📦 Will download {len(models_to_download)} model(s)")
    print(f"💾 Total estimated size: ~6-8 GB")
    print(f"📁 Cache directory: {cache_dir.absolute()}")
    
    # Ask for confirmation
    response = input("\n❓ Continue with download? (y/n): ")
    if response.lower() != 'y':
        print("❌ Download cancelled")
        sys.exit(0)
    
    # Download models
    success_count = 0
    for model in models_to_download:
        if download_model(model["repo_id"], str(cache_dir), model["name"]):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Downloaded {success_count}/{len(models_to_download)} models successfully")
    
    if success_count == len(models_to_download):
        print("\n🎉 All models downloaded! You can now use Enhanced Ghibli Processor.")
        print("\n💡 Note: Ghibli LoRA model is optional and can be added separately.")
    else:
        print("\n⚠️ Some models failed to download. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
