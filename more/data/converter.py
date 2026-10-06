"""Text-trajectory conversion utilities.

Converts between trajectory arrays and formatted text strings.
Example: np.array([[1., 2.], [3., 4.]]) <-> "[(1.00, 2.00), (3.00, 4.00)]"
"""

import ast
import re
import warnings
import numpy as np


# Default template settings
traj_prefix = "["
traj_suffix = "]"
traj_separator = ", "
coord_prefix = "("
coord_suffix = ")"
coord_template = "{:.2f}"
coord_separator = ", "


def change_template(template: dict):
    """Change the template for trajectory and coordinate conversion."""
    global traj_prefix, traj_suffix, traj_separator
    global coord_prefix, coord_suffix, coord_template, coord_separator

    for key in ("traj_prefix", "traj_suffix", "traj_separator",
                "coord_prefix", "coord_suffix", "coord_template", "coord_separator"):
        if key in template:
            globals()[key] = template[key]


def traj2text(trajectory: np.ndarray, prefix: str = "", suffix: str = "") -> str:
    """Convert a trajectory array (frame, dim) to a formatted string."""
    frame, dim = trajectory.shape
    coord_text_list = [
        coord_prefix
        + coord_separator.join(coord_template.format(trajectory[i, j]) for j in range(dim))
        + coord_suffix
        for i in range(frame)
    ]
    text = traj_prefix + traj_separator.join(coord_text_list) + traj_suffix
    return prefix + text + suffix


def text2traj(description: str, frame: int = 12, dim: int = 2):
    """Convert a formatted string to a trajectory array (frame, dim).

    Returns None if parsing fails. Tolerates ±1 frame difference.
    """
    error = False
    if len(description) == 0 or '[(' not in description:
        error = True

    # Try to repair truncated output: missing closing ')]'
    if not error and ')]' not in description:
        last_paren = description.rfind(')')
        if last_paren > description.find('[('):
            description = description[:last_paren + 1] + ']'
        else:
            error = True

    if not error:
        try:
            description_cleanup = description[description.find('[('):description.find(')]') + 2]
            description_cleanup = re.sub(
                r'[^0-9()\[\],.\-\n]', '',
                description_cleanup.replace('(', '[').replace(')', ']')
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                description_list = ast.literal_eval(description_cleanup)
        except Exception:
            error = True

    if not error and isinstance(description_list, list) and len(description_list) > 0 and \
       all(isinstance(description_list[i], (list, tuple)) and len(description_list[i]) == dim
           for i in range(len(description_list))):
        # Allow frame ±1 tolerance
        if len(description_list) == frame - 1:
            description_list.append(description_list[-1])
        elif len(description_list) == frame + 1:
            description_list = description_list[:frame]

        if len(description_list) == frame:
            try:
                traj = np.array(description_list)
            except Exception:
                error = True
        else:
            error = True
    else:
        error = True

    return None if error else traj


def batch_traj2txt(traj_list, prefix: str = "", suffix: str = ""):
    """Convert a list of trajectories to a list of strings."""
    return [traj2text(traj, prefix, suffix) for traj in traj_list]


def batch_text2traj(desc_list, frame: int = 12, dim: int = 2):
    """Convert a list of strings to a list of trajectories."""
    return [text2traj(desc, frame, dim) for desc in desc_list]
