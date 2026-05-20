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
    base_path =f'/oak/stanford/groups/mcovert/Instruments/Covert-lab-scope1/track_test'

    time_dict ='{"DUMM_gitg068_baeS_100225":{"018":{"1185":{"start": 65, "end": 85},"1260":{"start": 10, "end": 30}}},"DUMM_giTG060_064_121425":{"000":{"1343":{"start": 20, "end": 40}}}}'
    time_range_dict = json.loads(time_dict)
    # print(len(time_range_dict))
    phase_c_str = '0'
    fluor_c_str = '1'
	
    device = "automatic" 
    model = Trackastra.from_pretrained("general_2d_w_SAM2_features", device=device)
	


if __name__ == "__main__":
    run_track_astra()