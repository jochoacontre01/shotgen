#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np

# Add project root to sys.path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shotgen.evaluation import (
    contrast_to_noise_ratio,
    signal_to_noise_ratio,
    structural_similarity_index
)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NumPy arrays using Contrast-to-Noise Ratio (CNR), "
                    "Signal-to-Noise Ratio (SNR), and Structural Similarity Index (SSIM)."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the primary .npy array file to evaluate."
    )
    parser.add_argument(
        "-r", "--reference",
        type=str,
        default=None,
        help="Path to a reference .npy array file. If not provided, a noisy "
             "version of the primary array will be generated for comparison."
    )
    parser.add_argument(
        "--noise-snr",
        type=float,
        default=10.0,
        help="The target SNR (in dB) for generating the noisy version if no reference "
             "is provided. Default is 10.0."
    )
    
    args = parser.parse_args()
    
    # Load primary array
    if not os.path.exists(args.file):
        print(f"Error: Primary file '{args.file}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    try:
        arr = np.load(args.file)
    except Exception as e:
        print(f"Error: Failed to load '{args.file}'. {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loaded primary array from: {args.file}")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Range: [{arr.min()}, {arr.max()}]")
    print(f"  Mean:  {arr.mean()}")
    print(f"  Std:   {arr.std()}")
    print("-" * 50)
    
    if args.reference:
        # Load reference array
        if not os.path.exists(args.reference):
            print(f"Error: Reference file '{args.reference}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        try:
            ref_arr = np.load(args.reference)
        except Exception as e:
            print(f"Error: Failed to load reference '{args.reference}'. {e}", file=sys.stderr)
            sys.exit(1)
            
        print(f"Loaded reference array from: {args.reference}")
        print(f"  Shape: {ref_arr.shape}")
        print(f"  Dtype: {ref_arr.dtype}")
        print(f"  Range: [{ref_arr.min()}, {ref_arr.max()}]")
        print("-" * 50)
        
        # Verify shape matching
        if arr.shape != ref_arr.shape:
            print(f"Error: Shape mismatch. Primary {arr.shape} vs Reference {ref_arr.shape}", file=sys.stderr)
            sys.exit(1)
            
        # Compute metrics
        snr_val = signal_to_noise_ratio(arr, ref_arr)
        cnr_val = contrast_to_noise_ratio(arr, ref_arr=ref_arr)
        ssim_val = structural_similarity_index(arr, ref_arr)
        
        print("Evaluation Results (compared to reference array):")
        print(f"  Signal-to-Noise Ratio (SNR):      {snr_val:.4f} dB")
        print(f"  Contrast-to-Noise Ratio (CNR):    {cnr_val:.4f}")
        print(f"  Structural Similarity (SSIM):     {ssim_val:.4f}")
        
    else:
        # Generate noisy version for comparison
        print(f"No reference file provided. Generating noisy version with target SNR = {args.noise_snr} dB...")
        
        # We treat input 'arr' as the clean reference signal
        ref_arr = arr
        
        # Calculate standard deviation of noise needed to achieve target SNR
        # SNR = 10 * log10(signal_power / noise_power) => noise_power = signal_power / 10**(SNR/10)
        signal_power = np.mean(ref_arr**2)
        if signal_power == 0:
            print("Warning: Clean signal has zero power. Noise cannot be scaled relative to it.", file=sys.stderr)
            noise_std = 1.0
        else:
            noise_power = signal_power / (10.0 ** (args.noise_snr / 10.0))
            noise_std = np.sqrt(noise_power)
            
        noise = np.random.normal(0, noise_std, size=arr.shape).astype(arr.dtype)
        noisy_arr = ref_arr + noise
        
        # Compute metrics
        snr_val = signal_to_noise_ratio(noisy_arr, ref_arr)
        cnr_val = contrast_to_noise_ratio(noisy_arr, ref_arr=ref_arr)
        ssim_val = structural_similarity_index(noisy_arr, ref_arr)
        
        print("-" * 50)
        print("Evaluation Results (noisy test array vs clean primary array):")
        print(f"  Signal-to-Noise Ratio (SNR):      {snr_val:.4f} dB (Target: {args.noise_snr} dB)")
        print(f"  Contrast-to-Noise Ratio (CNR):    {cnr_val:.4f}")
        print(f"  Structural Similarity (SSIM):     {ssim_val:.4f}")
        
        # Also print metrics of the primary array on its own
        single_snr = signal_to_noise_ratio(arr)
        single_cnr = contrast_to_noise_ratio(arr)
        print("-" * 50)
        print("Self-Evaluation of Primary Array (estimated without reference):")
        print(f"  Estimated SNR:                    {single_snr:.4f} dB")
        print(f"  Estimated CNR:                    {single_cnr:.4f}")

if __name__ == "__main__":
    main()
