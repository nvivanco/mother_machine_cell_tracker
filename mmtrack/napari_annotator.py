import os

import numpy as np
import tifffile
import napari
from matplotlib import colormaps
from napari.utils.colormaps import label_colormap



def stack_to_kymograph(stack):
    """
    takes a numpyarray of an image in the format (t, y, x) and converts it into a kymograph
    """

    kymograph_gray = []
    for i in range(stack.shape[0]):
        frame = stack[i]
        if frame.ndim == 3:
            kymograph_gray.append(frame, axis=2)
        else:
            kymograph_gray.append(frame)
    
    kymograph = np.concatenate(kymograph_gray, axis =1)
    return kymograph

def kymograph_to_stack(kymograph, stack):
    """
    reversees stack_to_kymograph
    to ensure dimensions remain the same an input is the original stack in format (t,y,x) 
    """
    t = stack.shape[0]  
    y = stack.shape[1]  
    x = stack.shape[2]  
    
    reconstructed_stack = []
    for i in range(t):
        start_col = i * x
        end_col = (i + 1) * x
        frame = kymograph[:, start_col:end_col]  
        reconstructed_stack.append(frame)
    
    return np.array(reconstructed_stack)

def _add_frame_labels(viewer, phase_stack):
    """Overlay frame-number text at the top of each frame column in the kymograph."""
    num_frames = phase_stack.shape[0]
    frame_width = phase_stack.shape[2]

    coords = np.array(
        [[5.0, i * frame_width + frame_width / 2.0] for i in range(num_frames)]
    )
    text = {
        "string": [str(i) for i in range(num_frames)],
        "size": 7,
        "color": "white",
        "anchor": "center",
    }
    viewer.add_points(
        coords,
        text=text,
        size=1,
        face_color="transparent",
        border_color="transparent",
        name="frame_numbers",
    )


def run_annotator(phase_path, mask_path, FOV, peak_id):
    """
    this function sets up napari to edit labels from mmmct pipeline

    Args:
        phase_path: Path to the phase image stack (T,Y,X).
        mask_path:  Path to the auto-generated mask stack (T,Y,X).

    Returns:
        Path to the saved corrected mask file.
    """

    phase_stack = tifffile.imread(phase_path)
    mask_stack = tifffile.imread(mask_path)


    viewer = napari.Viewer()
    viewer.add_image(stack_to_kymograph(phase_stack), name="phase", colormap="gray")
    num_colors = 20
    tab20_colors = label_colormap(num_colors, seed=0.5)

    label_layer = viewer.add_labels(
        stack_to_kymograph(mask_stack).astype("uint32"),
        name="segmentation",
        colormap=tab20_colors
    )

    _add_frame_labels(viewer, phase_stack)

    print("    ---napari open. Edit labels, then close the window to save.")
    napari.run()

    corrected_kymograph = label_layer.data
    corrected = kymograph_to_stack(corrected_kymograph, mask_stack)

    corrections_dir = os.path.join(os.path.dirname(phase_path), "napari_corrections")
    os.makedirs(corrections_dir, exist_ok=True)

    output_path = os.path.join(corrections_dir, f'{FOV}_{peak_id}_corrected.tif')
    if os.path.exists(output_path):
        response = input(f"    ---File already exists: {output_path}\n    --- Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("    ---Save cancelled.")
            return output_path
    tifffile.imwrite(output_path, corrected.astype(mask_stack.dtype))
    print(f"    ---Corrected masks saved to: {output_path}")

    return output_path
