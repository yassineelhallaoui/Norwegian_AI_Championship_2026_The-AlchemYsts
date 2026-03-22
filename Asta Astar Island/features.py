"""Feature extraction for per-cell surrogate modeling."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

from constants import INTERNAL_TERRAIN_CODES, INTERNAL_TO_PREDICTION_CLASS


NEIGHBORHOOD_KERNELS = [
    np.ones((3, 3), dtype=float),
    np.ones((5, 5), dtype=float),
    np.ones((7, 7), dtype=float),
    np.ones((15, 15), dtype=float),
]

CARDINAL_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _distance_to(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(~mask)


def build_feature_tensor(initial_state: dict) -> np.ndarray:
    """Return a HxWxF tensor from the initial map state."""

    grid = np.asarray(initial_state["grid"], dtype=int)
    height, width = grid.shape

    class_grid = np.vectorize(INTERNAL_TO_PREDICTION_CLASS.get)(grid)
    x_coords = np.tile(np.arange(width), (height, 1))
    y_coords = np.tile(np.arange(height)[:, None], (1, width))

    terrain_masks = [
        grid == 10,
        grid == 4,
        grid == 5,
        np.isin(grid, [1, 2]),
        grid == 2,
        grid == 11,
        grid != 10,
    ]

    features: list[np.ndarray] = []
    for code in INTERNAL_TERRAIN_CODES:
        features.append((grid == code).astype(float))

    for class_index in range(6):
        features.append((class_grid == class_index).astype(float))

    for mask in terrain_masks:
        mask_float = mask.astype(float)
        for kernel in NEIGHBORHOOD_KERNELS:
            features.append(convolve(mask_float, kernel, mode="constant", cval=0.0))

    ocean_mask = grid == 10
    ocean_adjacency = convolve(ocean_mask.astype(float), CARDINAL_KERNEL, mode="constant", cval=0.0)
    features.append((ocean_adjacency > 0).astype(float))
    features.append(ocean_adjacency)

    # New V2 Features
    forest_mask = grid == 4
    port_mask = np.isin(grid, [1, 2])
    
    # Food Potential (Forest density)
    food_potential = convolve(forest_mask.astype(float), np.ones((7, 7)), mode="constant", cval=0.0)
    features.append(food_potential)

    # Fjord / Coastline Density
    fjord_density = convolve(ocean_mask.astype(float), np.ones((7, 7)), mode="constant", cval=1.0)
    features.append(fjord_density)
    
    # Trade and Expansion Pressure
    trade_density = convolve(port_mask.astype(float), np.ones((15, 15)), mode="constant", cval=0.0)
    features.append(trade_density)

    for mask in [
        ocean_mask,
        grid == 4,
        grid == 5,
        np.isin(grid, [1, 2]),
        grid == 2,
        grid == 11,
    ]:
        features.append(_distance_to(mask))

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radial_scale = np.sqrt(center_x**2 + center_y**2)
    edge_distance = np.minimum.reduce([x_coords, y_coords, width - 1 - x_coords, height - 1 - y_coords])

    features.extend(
        [
            x_coords / max(width - 1, 1),
            y_coords / max(height - 1, 1),
            (x_coords - center_x) / max(center_x, 1.0),
            (y_coords - center_y) / max(center_y, 1.0),
            np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) / max(radial_scale, 1.0),
            edge_distance / max(min(height, width) / 2.0, 1.0),
        ]
    )

    return np.stack(features, axis=-1)


def build_feature_matrix(initial_state: dict) -> np.ndarray:
    tensor = build_feature_tensor(initial_state)
    return tensor.reshape(-1, tensor.shape[-1])
