"""
Utility function to load corrected kymograph and verify labels are unique.
"""

import numpy as np
import tifffile
from scipy import ndimage


def check_labels_within_frames(mask, min_region_size=20):
    """
    Check for duplicate labels and unusually large labels within each frame.
    
    Parameters
    ----------
    mask : np.ndarray
        3D mask array (frames, height, width)
    min_region_size : int
        Minimum pixel count to consider a region as a real cell (default: 20)
    """
    definite_duplicates = []
    large_labels = []
    
    for frame_idx in range(mask.shape[0]):
        frame = mask[frame_idx]
        labels_in_frame = np.unique(frame)
        labels_in_frame = labels_in_frame[labels_in_frame != 0]
        
        if len(labels_in_frame) == 0:
            continue
        
        # Get sizes of each label in this frame
        sizes = {label: np.sum(frame == label) for label in labels_in_frame}
        median_size = np.median(list(sizes.values()))
        
        for label in labels_in_frame:
            # Check 1: Multiple separate regions larger than min_region_size?
            binary = (frame == label)
            labeled, num_regions = ndimage.label(binary)
            
            # Count only regions above minimum size
            real_regions = 0
            for i in range(1, num_regions + 1):
                if np.sum(labeled == i) >= min_region_size:
                    real_regions += 1
            
            if real_regions > 1:
                definite_duplicates.append((frame_idx, label, real_regions))
            
            # Check 2: Unusually large?
            if sizes[label] > 2 * median_size:
                large_labels.append((frame_idx, label, sizes[label], median_size))
    
    # Report
    if definite_duplicates:
        print(f"  ❌ Duplicate labels found (same label used for separate cells):")
        for frame, label, count in definite_duplicates:
            print(f"      Frame {frame}: label {label} appears in {count} separate regions")
    else:
        print(f"  ✅ No duplicate labels within frames")
    
    if large_labels:
        print(f"  ⚠️  Large labels (could be a dividing cell OR adjacent cells with same label):")
        for frame, label, size, median in large_labels[:10]:
            print(f"      Frame {frame}: label {label} is {size/median:.1f}x median size")
    else:
        print(f"  ✅ No unusually large labels")
    
    return definite_duplicates, large_labels


def load_and_verify_corrected_mask(corrected_path):
    """
    Load a corrected segmentation mask and verify all cell labels are unique.
    
    Parameters
    ----------
    corrected_path : str
        Path to the corrected .tif mask file
        
    Returns
    -------
    mask : np.ndarray
        The loaded mask array (frames, height, width)
    n_cells : int
        Number of unique cell labels (excluding background)
    is_valid : bool
        True if all labels are unique within frames
    """
    mask = tifffile.imread(corrected_path)
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    n_cells = len(unique_labels)
    max_label = unique_labels.max() if len(unique_labels) > 0 else 0
    
    print(f"Loaded: {corrected_path}")
    print(f"  Shape: {mask.shape}")
    print(f"  Unique cell labels: {n_cells}")
    print(f"  Label range: 1 to {max_label}")
    
    # Run within-frame checks
    print(f"\n--- Label checks ---")
    duplicates_within, large = check_labels_within_frames(mask)
    
    is_valid = len(duplicates_within) == 0
    
    return mask, n_cells, is_valid


def mask_to_kymograph(mask):
    """
    Convert a 3D mask stack (frames, H, W) to a 2D kymograph (H, frames*W).
    """
    return np.hstack([mask[i] for i in range(mask.shape[0])])


def view_corrected_mask(corrected_path):
    """
    Load corrected mask, verify labels, and display as 3D stack in napari.
    Shows frame slider for easy navigation.
    """
    import napari
    
    mask, n_cells, is_valid = load_and_verify_corrected_mask(corrected_path)
    
    viewer = napari.Viewer()
    viewer.add_labels(mask, name=f'corrected_mask ({n_cells} cells)')
    napari.run()


def view_corrected_kymograph(corrected_path):
    """
    Load corrected mask, verify labels, and display as 2D kymograph in napari.
    """
    import napari
    
    mask, n_cells, is_valid = load_and_verify_corrected_mask(corrected_path)
    kymo = mask_to_kymograph(mask)
    
    viewer = napari.Viewer()
    viewer.add_labels(kymo, name=f'corrected_kymo ({n_cells} cells)')
    napari.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to corrected mask: ")
    
    view_corrected_mask(path)
