from .board import Board
from .pieces import get_random_piece, rotate, SHAPES
import copy

class TetrisEnv:
    def __init__(self):
        self.board = Board()
        self.current_shape = get_random_piece()
        self.next_shape = get_random_piece()

    def reset(self):
        self.board.reset()
        self.current_shape = get_random_piece()
        self.next_shape = get_random_piece()
        return self.get_state_features()

    def simulate_step(self, action):
        env_copy = copy.deepcopy(self)
        return env_copy.step(action)

    def step(self, action):
        """
        action = (rotation, column)
        """

        rotation, column = action

        shape, _ = self.current_shape
        shape = copy.deepcopy(shape)

        # Apply rotation
        for _ in range(rotation):
            shape = rotate(shape)

        x = column
        y = 0

        # Drop piece
        while self.board.is_valid_position(shape, x, y + 1):
            y += 1

        # Invalid placement → game over
        if not self.board.is_valid_position(shape, x, y):
            return self.get_state_features(), -100, True

        # Lock piece
        self.board.place_piece(shape, x, y)

        # Clear lines
        lines_cleared = self.board.clear_lines()

        # Reward good placements and penalize bad board shape
        features = self.get_state_features()
        aggregate_height = features[10]
        holes = features[11]
        bumpiness = features[12]
        reward = lines_cleared * 40 - holes * 5 - bumpiness * 0.8 - aggregate_height * 0.15

        # Spawn next piece
        self.current_shape = self.next_shape
        self.next_shape = get_random_piece()

        # Check game over
        if not self.board.is_valid_position(self.current_shape[0], 3, 0):
            return self.get_state_features(), reward, True

        return self.get_state_features(), reward, False


    def get_valid_actions(self):
        valid_actions = []

        for rotation in range(4):
            shape, _ = self.current_shape
            shape = copy.deepcopy(shape)

            for _ in range(rotation):
                shape = rotate(shape)

            shape_width = len(shape[0])

            for column in range(self.board.width - shape_width + 1):

                x = column
                y = 0

                while self.board.is_valid_position(shape, x, y + 1):
                    y += 1

                if self.board.is_valid_position(shape, x, y):
                    valid_actions.append((rotation, column))

        return valid_actions


    def get_state_features(self):
        grid = self.board.grid
        width = self.board.width
        height = self.board.height

        heights = []
        holes = 0

        for col in range(width):
            column_height = 0
            block_found = False

            for row in range(height):
                if grid[row][col] == 1:
                    if not block_found:
                        column_height = height - row
                        block_found = True
                elif block_found:
                    holes += 1

            heights.append(column_height)

        aggregate_height = sum(heights)

        bumpiness = 0
        for i in range(width - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        current_id = self.current_shape[1]
        next_id = self.next_shape[1]
        current_one_hot = [1 if i == current_id else 0 for i in range(len(SHAPES))]
        next_one_hot = [1 if i == next_id else 0 for i in range(len(SHAPES))]

        return heights + [aggregate_height, holes, bumpiness] + current_one_hot + next_one_hot