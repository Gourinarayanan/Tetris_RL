class Board:
    def __init__(self):
        self.width = 10
        self.height = 20
        self.grid = [[0] * self.width for _ in range(self.height)]

    def reset(self):
        self.grid = [[0] * self.width for _ in range(self.height)]

    def is_valid_position(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = x + offset_x
                    new_y = y + offset_y

                    if (
                        new_x < 0 or new_x >= self.width or
                        new_y >= self.height or
                        (new_y >= 0 and self.grid[new_y][new_x])
                    ):
                        return False
        return True

    def place_piece(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[y + offset_y][x + offset_x] = 1

    def clear_lines(self):
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        lines_cleared = self.height - len(new_grid)

        while len(new_grid) < self.height:
            new_grid.insert(0, [0] * self.width)

        self.grid = new_grid
        return lines_cleared