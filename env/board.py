class Board:
    WIDTH  = 10
    HEIGHT = 20

    def __init__(self):
        self.width  = self.WIDTH
        self.height = self.HEIGHT
        self.reset()

    def reset(self):
        # grid stores piece color-index+1 (0 = empty)
        self.grid = [[0] * self.width for _ in range(self.height)]

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def is_valid_position(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    nx = x + offset_x
                    ny = y + offset_y
                    if nx < 0 or nx >= self.width:
                        return False
                    if ny >= self.height:
                        return False
                    if ny >= 0 and self.grid[ny][nx]:
                        return False
        return True

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------
    def place_piece(self, shape, offset_x, offset_y, color_id=1):
        """Lock piece onto the grid.  color_id is piece index + 1."""
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[y + offset_y][x + offset_x] = color_id

    # ------------------------------------------------------------------
    # Line clearing – returns number of lines cleared
    # ------------------------------------------------------------------
    def clear_lines(self):
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]
        lines_cleared = self.height - len(new_grid)
        while len(new_grid) < self.height:
            new_grid.insert(0, [0] * self.width)
        self.grid = new_grid
        return lines_cleared

    # ------------------------------------------------------------------
    # Analytics helpers (used by the AI feature extractor)
    # ------------------------------------------------------------------
    def column_heights(self):
        heights = []
        for col in range(self.width):
            h = 0
            for row in range(self.height):
                if self.grid[row][col]:
                    h = self.height - row
                    break
            heights.append(h)
        return heights

    def count_holes(self):
        holes = 0
        for col in range(self.width):
            block_found = False
            for row in range(self.height):
                if self.grid[row][col]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def bumpiness(self, heights=None):
        if heights is None:
            heights = self.column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(self.width - 1))

    def count_wells(self):
        """Deep wells are extra-penalised columns lower than both neighbours."""
        heights = self.column_heights()
        well_sum = 0
        for i in range(self.width):
            left  = heights[i - 1] if i > 0                  else 99
            right = heights[i + 1] if i < self.width - 1     else 99
            depth = min(left, right) - heights[i]
            if depth > 0:
                well_sum += depth
        return well_sum

    def row_transitions(self):
        """Count horizontal empty↔filled transitions."""
        count = 0
        for row in self.grid:
            prev = 1
            for cell in row:
                occupied = 1 if cell else 0
                if occupied != prev:
                    count += 1
                prev = occupied
            if prev == 0:
                count += 1
        return count

    def col_transitions(self):
        """Count vertical empty↔filled transitions."""
        count = 0
        for col in range(self.width):
            prev = 1
            for row in range(self.height):
                occupied = 1 if self.grid[row][col] else 0
                if occupied != prev:
                    count += 1
                prev = occupied
        return count