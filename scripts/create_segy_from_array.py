#!/usr/bin/env python3
"""
Convert an array or .npy file into a SEG-Y velocity model compliant with shotgen.
This script acts as a CLI utility and wrapper for shotgen.io.convert_npy_to_segy.

Example:
    python scripts/create_segy_from_array.py assets/fpb.npy --dx 50 --dz 50
"""

from npy_to_segy import main

if __name__ == "__main__":
    main()
