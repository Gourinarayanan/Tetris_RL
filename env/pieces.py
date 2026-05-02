import random
import copy

SHAPES = [
    [[1,1,1,1]],
    [[1,1],
     [1,1]],
    [[0,1,0],
     [1,1,1]],
    [[1,1,0],
     [0,1,1]],
    [[0,1,1],
     [1,1,0]],
    [[1,0,0],
     [1,1,1]],
    [[0,0,1],
     [1,1,1]]
]

def rotate(shape):
    return [list(row) for row in zip(*shape[::-1])]

def get_random_piece():
    idx = random.randrange(len(SHAPES))
    return copy.deepcopy(SHAPES[idx]), idx