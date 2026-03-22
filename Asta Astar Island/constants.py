"""Shared constants for Astar Island."""

INTERNAL_TERRAIN_CODES = [0, 1, 2, 3, 4, 5, 10, 11]
PREDICTION_CLASS_COUNT = 6

PREDICTION_CLASSES = [
    "empty",
    "settlement",
    "port",
    "ruin",
    "forest",
    "mountain",
]

INTERNAL_TO_PREDICTION_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    10: 0,
    11: 0,
}

EMPTY_CLASS = 0
MOUNTAIN_CLASS = 5
VIEWPORT_SIZE = 15
PROBABILITY_FLOOR = 0.01
