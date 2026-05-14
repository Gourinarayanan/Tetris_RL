import random
import copy

# All 7 standard Tetris pieces (tetrominoes)
SHAPES = [
    [[1, 1, 1, 1]],           # I-piece  (index 0)
    [[1, 1],                   # O-piece  (index 1)
     [1, 1]],
    [[0, 1, 0],                # T-piece  (index 2)
     [1, 1, 1]],
    [[1, 1, 0],                # S-piece  (index 3)
     [0, 1, 1]],
    [[0, 1, 1],                # Z-piece  (index 4)
     [1, 1, 0]],
    [[1, 0, 0],                # J-piece  (index 5)
     [1, 1, 1]],
    [[0, 0, 1],                # L-piece  (index 6)
     [1, 1, 1]],
]

# Vibrant, distinct color per piece (RGB)
PIECE_COLORS = [
    (0,   240, 240),   # I – Cyan
    (240, 240,   0),   # O – Yellow
    (160,   0, 240),   # T – Purple
    (0,   240,   0),   # S – Green
    (240,   0,   0),   # Z – Red
    (0,    80, 240),   # J – Blue
    (240, 160,   0),   # L – Orange
]


def rotate(shape):
    """90° clockwise rotation."""
    return [list(row) for row in zip(*shape[::-1])]


def get_all_rotations(shape):
    """Return list of up to 4 unique rotation states."""
    rotations = []
    seen = set()
    current = shape
    for _ in range(4):
        key = tuple(tuple(r) for r in current)
        if key not in seen:
            seen.add(key)
            rotations.append(copy.deepcopy(current))
        current = rotate(current)
    return rotations


# 7-bag randomizer for fair piece distribution (like real Tetris)
_bag: list[int] = []

def get_random_piece():
    global _bag
    if not _bag:
        _bag = list(range(len(SHAPES)))
        random.shuffle(_bag)
    idx = _bag.pop()
    return copy.deepcopy(SHAPES[idx]), idx