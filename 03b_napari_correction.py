"""
Stage 3b: Interactive napari correction of auto-generated segmentation masks.

Run this after 03_cell_segmentation.py. It opens napari for each trench so
the user can paint / erase / fill labels with napari's built-in tools.
Corrected masks are saved to a 'napari_corrections' directory next to the
original mask file, under the same filename. The original masks are never
modified.

Usage:
    poetry run python 03b_napari_correction.py \
        --base-dir /path/to/data \
        --exp-dict '{"exp_dir": {"FOV": ["trench_id", ...]}}' \
        [--phase-channel 0]
"""

import argparse
import json
import os
import tifffile

def run_napari_correction(base_dir, exp_dict_json, phase_c_str):
    try:
        seg_FOV_dict = json.loads(exp_dict_json)
    except json.JSONDecodeError:
        print("ERROR: Could not parse the experiment dictionary. Ensure it is valid JSON.")
        return

    from mmtrack.napari_annotator import run_annotator

    for folder, fov_dict in seg_FOV_dict.items():
        print(f"\nProcessing Experiment: {folder}")
        for fov_id, ana_peak_ids in fov_dict.items():
            print(f"  FOV: {fov_id}, Trenches: {ana_peak_ids}")

            for peak_id in ana_peak_ids:
                base_file_path = os.path.join(
                    base_dir, folder, "hyperstacked", "drift_corrected",
                    "rotated", "mm_channels", "subtracted"
                )

                path_to_phase_stack = os.path.join(
                    base_file_path,
                    f"subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif"
                )
                mask_path = os.path.join(
                    base_file_path,
                    f"mm3_segmented_subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_c_str}.tif"
                )

                if not os.path.exists(mask_path):
                    print(f"  WARNING: mask file not found, skipping: {mask_path}")
                    continue
                if not os.path.exists(path_to_phase_stack):
                    print(f"  WARNING: phase stack not found, skipping: {path_to_phase_stack}")
                    continue



                print(f"  -> Trench {peak_id}: opening napari for correction...")
                run_annotator(path_to_phase_stack, mask_path, fov_id, peak_id)

    print("\nStage 3b (napari correction) complete.")
    print("Corrected masks are in 'napari_corrections/' next to each original mask file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3b: Interactive napari correction of segmentation masks."
    )
    parser.add_argument(
        "--base-dir", required=True, type=str,
        help="Base path containing all experiment folders."
    )
    parser.add_argument(
        "--exp-dict", required=True, type=str,
        help='JSON string defining experiments/FOVs/trenches. '
             'Format: {"exp_dir": {"FOV": ["peak_id", ...]}}.'
    )
    parser.add_argument(
        "--phase-channel", type=str, default="0",
        help="Phase channel index as string. Default: '0'."
    )

    args = parser.parse_args()

    run_napari_correction(
        base_dir=args.base_dir,
        exp_dict_json=args.exp_dict,
        phase_c_str=args.phase_channel,
    )
