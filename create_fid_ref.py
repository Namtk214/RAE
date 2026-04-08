"""Script to compute FID reference statistics (mu, sigma) for a dataset."""
import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
import torch
from tqdm import tqdm
from data import build_dataloader

from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

def main():
    parser = argparse.ArgumentParser(description="Compute Validation FID statistics (mu, sigma) for a given dataset.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--out-path", type=str, required=True, help="Output NPZ path (e.g. fid_ref_stat.npz)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of images to process.")
    parser.add_argument("--dataset-type", type=str, default="tfds", choices=["imagefolder", "tfds"])
    parser.add_argument("--tfds-name", type=str, default="celebahq256")
    parser.add_argument("--tfds-builder-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="validation", help="Split to compute stats on (validation/train/all)")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data_path}")
    ds, steps = build_dataloader(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        split=args.split,
        tfds_name=args.tfds_name,
        tfds_builder_dir=args.tfds_builder_dir,
    )

    print(f"Loading InceptionV3 model on {args.device}...")
    fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=['2048']).to(args.device).eval()

    all_features = []
    processed = 0

    print(f"Extracting Inception features up to {args.num_samples} samples...")
    # Wrap in tqdm
    pbar = tqdm(total=args.num_samples)
    
    with torch.no_grad():
        for step_data in ds:
            if processed >= args.num_samples:
                break
                
            # step_data["image"] is typically (B, H, W, C) float [0, 1] or uint8
            # Convert to numpy first
            batch_images = np.array(step_data["image"])
            
            # Format to NHWC uint8 [0, 255] if necessary
            if batch_images.dtype != np.uint8:
                if batch_images.max() <= 2.0:
                    batch_images = batch_images * 255.0
                batch_images = np.clip(batch_images, 0, 255).astype(np.uint8)
                
            # PyTorch expects NCHW
            if batch_images.shape[-1] == 3:
                batch_images = np.transpose(batch_images, (0, 3, 1, 2))
                
            batch_tensor = torch.from_numpy(batch_images).to(device=args.device, dtype=torch.uint8)
            
            # Extract features
            feats = fe(batch_tensor)[0]
            all_features.append(feats.cpu().numpy())
            
            n_batch = batch_images.shape[0]
            processed += n_batch
            pbar.update(n_batch)

    pbar.close()
    
    if processed == 0:
        raise RuntimeError("No images processed! Check dataloader and data-path.")

    print("Concatenating features...")
    # Concatenate all lists into single matrix (N, 2048)
    all_features_np = np.concatenate(all_features, axis=0)[:args.num_samples]
    
    print("Computing Mean (mu) and Covariance (sigma)...")
    mu = np.mean(all_features_np, axis=0)
    sigma = np.cov(all_features_np, rowvar=False)

    print(f"Saving statistics to {args.out_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    np.savez(args.out_path, mu=mu, sigma=sigma)
    print("Done!")

if __name__ == "__main__":
    main()
