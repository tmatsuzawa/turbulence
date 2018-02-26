# Visualizations for experimental movies	

This repository contains code to make data from the gyroscope experiments into movies with circles that are colored
either by phase or amplitude.

Currently only works with cine files.  Can also be used to just plot circles representing amplitudes with no background
of the actual experimental video.

## Overview of functions

### `get_frame_points_and_time`: 
This function gets the gyroscopes'
*parameters*:
* data - The data loaded from the tracked hdf5 com_data file.  This should be listed with the gyroscope numbers being the keys
* keys - the keys from the hdf5 files.  You can get this by saying data.keys(), but I just pass the keys here.
* i - Integer. The frame number that you want the data for

*returns*
* time - Float. The time of frame i
* pts - Float array.  Number of gyroscopes x 2. The gyroscopes' positions in frame i

### `get_average_positions`: 

*parameters*:
* data - The data loaded from the tracked hdf5 com_data file.  This should be listed with the gyroscope numbers being the keys
* keys - the keys from the hdf5 files.  You can get this by saying data.keys(), but I just pass the keys here.
* range - range of frames over which to perform the average. Input this as [lower_bound, upper_bound]

*returns*
* xavg - Float array of size Number of gyros.  Average x positions of gyroscopes over interval
* yavg - Float array of size Number of gyros.  Average y positions of gyroscopes over interval

### `plot_on_frame`:
Plots the data from the file on the movie frame and saves.

*parameters*:
* frame - The data from the frame you are plotting.  This is what cirlces will be drawn over.
* time - The timestamp of the frame you are plotting
* pts - The current positions of the gyros in this frame
* avg - The average positions of the gyros for this time interval
* plotting_mag - Number by which to multiply displacement for plotting. Optional argument.  Is set to 1 by default.
* save_name - Name of saved frame.  If you would like to save in a different directory, the entire path must be passed here.
* color_by - Choice of whether to color the circles by phase or amplitude
* cmap - color map for color_by parameter.

*returns*
* NONE

### `adjust_frame`:
Adjusts the brightness of the frame as prescribed by maximum and minimum values

*parameters*:
* current_frame - Data array of current frame image data.
* min_value - Minimum value for brightness adjustment.
* max_value - Maximum value for brightness adjustment.

*returns*
* current_frame - the adjusted data for the current frame.

### `make_frames`: 
This function will make all the frames.  First round goes mod 16, then 8, then 4, then 2, then 1.

*parameters*:
* root_dir - root directory where the com_data hdf5 file should be
* video_path - path for the cine video
* fn_for_movie - file name for movie that will eventually be produced
* color_by - color by phase or amplitude. (string)
* cmap - what color map you'd like to use.

*returns*
* NONE