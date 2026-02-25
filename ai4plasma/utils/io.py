"""Input/Output utilities for AI4Plasma.

This module provides utility functions for handling input and output operations
in the AI4Plasma project, with a focus on JSON configuration file reading and
image-to-GIF conversion for training visualization.

IO Functions
------------
- `read_json`: Load and parse JSON configuration files
- `img2gif`: Convert a sequence of images into an animated GIF
"""

import json
import imageio.v2 as imageio


def read_json(json_file):
    """Read and parse a JSON configuration file.
    
    Loads a JSON file and returns its content as a Python dictionary.
    Provides detailed error messages for common file operation failures.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file to be read.
    
    Returns
    -------
    dict
        Parsed content of the JSON file as a dictionary.
    
    Raises
    ------
    FileNotFoundError
        If the specified JSON file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON format.
    
    Examples
    --------
    >>> config = read_json('config.json')
    >>> learning_rate = config['train']['lr']
    """
    try:
        with open(json_file, 'r') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_file} does not exist.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Failed to decode JSON from the file.", doc=json_file, pos=0)
    
    return cfg


def img2gif(img_file_list, gif_file, duration=500, loop=0):
    """Convert a sequence of images into an animated GIF file.
    
    Reads a list of image files and combines them into a single animated GIF
    with configurable frame timing and loop behavior. Supports all image formats
    supported by the imageio library.
    
    Parameters
    ----------
    img_file_list : list of str
        List of file paths to input images. Images will be combined in the
        order provided.
    gif_file : str
        File path where the output GIF will be saved.
    duration : int, optional
        Duration in milliseconds for each frame. Default is 500.
    loop : int, optional
        Number of times the GIF should loop. 0 means infinite loop. Default is 0.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the img_file_list is empty.
    FileNotFoundError
        If any of the specified image files do not exist.
    """
    if not img_file_list:
        raise ValueError("The img_file_list is empty.")
    
    frames = []
    for img_file in img_file_list:
        try:
            frames.append(imageio.imread(img_file))
        except FileNotFoundError:
            raise FileNotFoundError(f"The image file {img_file} does not exist.")
    
    imageio.mimsave(gif_file, frames, 'GIF', duration=duration, loop=loop)

