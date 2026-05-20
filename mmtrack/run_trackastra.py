import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff
from trackastra.data import example_data_bacteria
import tifffile
import napari
import json
import pandas as np
import numpy as np


def get_tiff_frame_count(file_path):
	"""
	Reads a TIFF file using imread and returns the index of the last time frame (T - 1).
	NOTE: This loads the entire file into memory.
	"""
	try:
		# Load the entire image stack into memory
		img_stack = tifffile.imread(file_path)

		shape = img_stack.shape

		# Assume the time axis (T) is the first dimension
		if len(shape) >= 3:
			# The last index is T - 1
			return shape[0] - 1
		else:
			# Single 2D image
			return 0

	except FileNotFoundError:
		print(f"Error: TIFF file not found at {file_path}")
		return 0
	except Exception as e:
		print(f"Error reading TIFF file {file_path}: {e}")
		return 0
	
def plot_trackastra_kymograph(imgs, ctc_masks, napari_tracks, napari_tracks_graph):
    kymo_imgs = imgs.transpose(1, 0, 2).reshape(382, -1)  # (382, 1800)
    kymo_masks = ctc_masks.transpose(1, 0, 2).reshape(382, -1)

    kymo_tracks = napari_tracks.copy()
    new_x = napari_tracks[:, 1] * 20 + napari_tracks[:, 3]
    kymo_tracks = np.column_stack([
        napari_tracks[:, 0],  # track_id
        np.zeros(len(napari_tracks)),  # dummy time (all in same frame)
        napari_tracks[:, 2],  # y stays the same
        new_x  # new x position
])

def run_track_astra():
    path_all_lineages_df = '/Users/adrianjuarez/Documents/Covert_lab/Projects/Operon/tracked_all_cell_data_aggregate_032626.pkl'
    path_all_cell_data_df = '/Users/adrianjuarez/Documents/Covert_lab/Projects/Operon/all_cell_data_aggregate_032626.pkl'
    all_lineages_df = pd.read_pickle(path_all_lineages_df)
    all_cell_data_df = pd.read_pickle(path_all_cell_data_df)

    experiment = 'DUMM_gitg068_baeS_100225'
    base_path =f'/Volumes/mcovert/Instruments/Covert-lab-scope1/subgen_processed_data/{experiment}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted'
    path_to_phase_stack_dir=f'{base_path}'
    path_to_labels_stack_dir =f'{base_path}/mask_kymos'
    phase_list = os.listdir(path_to_phase_stack_dir)
    mask_list =os.listdir(path_to_labels_stack_dir)

    path_to_mask = f'{base_path}/mm3_segmented_subtracted_FOV_018_region_1185_c_0.tif'
    path_to_phase = f'{base_path}/subtracted_FOV_018_region_1185_c_0.tif'
    imgs=tifffile.imread(path_to_phase)
    masks=tifffile.imread(path_to_mask)
    

    with open("napari_tracks_graph.json") as f:
        napari_tracks_graph = json.load(f)
        napari_tracks_graph = {int(k): int(v) for k, v in napari_tracks_graph.items()}
    napari_tracks = np.load("napari_tracks.npy")
    ctc_masks = np.load("ctc_masks.npy")
    v = napari.Viewer()
    v.add_image(imgs)
    v.add_labels(ctc_masks)
    v.add_tracks(data=napari_tracks, graph=napari_tracks_graph)

if __name__ == "__main__":
    run_track_astra()