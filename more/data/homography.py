"""Image/world coordinate conversion via homography matrices."""

import numpy as np
import torch


def image2world(coord, H):
    """Convert image coordinates to world coordinates.

    Args:
        coord: Image coordinates, shape (..., 2). numpy or torch.
        H: Homography matrix, shape (3, 3). Same type as coord.

    Returns:
        World coordinates with same shape and type as coord.
    """
    assert coord.shape[-1] == 2
    assert H.shape == (3, 3)

    shape = coord.shape
    coord = coord.reshape(-1, 2)

    if isinstance(coord, np.ndarray):
        x, y = coord[..., 0], coord[..., 1]
        world = (H @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
        world = world / world[..., [2]]
        world = world[..., :2]
    elif isinstance(coord, torch.Tensor):
        x, y = coord[..., 0], coord[..., 1]
        world = (H @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
        world = world / world[..., [2]]
        world = world[..., :2]
    else:
        raise NotImplementedError(f"Unsupported type: {type(coord)}")

    return world.reshape(shape)


def world2image(coord, H):
    """Convert world coordinates to image coordinates.

    Args:
        coord: World coordinates, shape (..., 2). numpy or torch.
        H: Homography matrix, shape (3, 3). Same type as coord.

    Returns:
        Image coordinates with same shape and type as coord.
    """
    assert coord.shape[-1] == 2
    assert H.shape == (3, 3)

    shape = coord.shape
    coord = coord.reshape(-1, 2)

    if isinstance(coord, np.ndarray):
        x, y = coord[..., 0], coord[..., 1]
        image = (np.linalg.inv(H) @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
        image = image / image[..., [2]]
        image = image[..., :2]
    elif isinstance(coord, torch.Tensor):
        x, y = coord[..., 0], coord[..., 1]
        image = (torch.linalg.inv(H) @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
        image = image / image[..., [2]]
        image = image[..., :2]
    else:
        raise NotImplementedError(f"Unsupported type: {type(coord)}")

    return image.reshape(shape)


def generate_homography(shift_w: float = 0, shift_h: float = 0,
                        rotate: float = 0, scale: float = 1):
    """Generate a homography matrix with translation, rotation, and scale."""
    H = np.eye(3)
    H[0, 2] = shift_w
    H[1, 2] = shift_h
    H[2, 2] = scale

    if rotate != 0:
        R = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate),  np.cos(rotate), 0],
            [0, 0, 1]
        ])
        H = H @ R

    return H
