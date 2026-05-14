import copy
from .board import Board
from .pieces import get_random_piece, rotate, SHAPES, PIECE_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# Feature vector layout  (length = STATE_DIM  →  27 values)
# ─────────────────────────────────────────────────────────────────────────────
#   [0..9]   column heights           (10)
#   [10]     aggregate height         (1)
#   [11]     holes                    (1)
#   [12]     bumpiness                (1)
#   [13]     wells                    (1)
#   [14]     row transitions          (1)
#   [15]     col transitions          (1)
#   [16..22] current piece one-hot   (7)
#   [23..29] next piece one-hot      (7)   → total = 30
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM   = 30          # must match DQN input_dim in neural.py
NUM_PIECES  = len(SHAPES)


class TetrisEnv:
    def __init__(self):
        self.board         = Board()
        self.current_shape = get_random_piece()
        self.next_shape    = get_random_piece()
        self._total_lines  = 0

    # -------------------------------------------------------------------------
    def reset(self):
        self.board         = Board()
        self.current_shape = get_random_piece()
        self.next_shape    = get_random_piece()
        self._total_lines  = 0
        return self.get_state_features()

    # -------------------------------------------------------------------------
    # Shallow simulation copy (fast – avoids deepcopy of the whole object)
    # -------------------------------------------------------------------------
    def _fast_copy(self):
        env            = TetrisEnv.__new__(TetrisEnv)
        env.board      = Board()
        env.board.grid = [row[:] for row in self.board.grid]
        env.current_shape = self.current_shape
        env.next_shape    = self.next_shape
        env._total_lines  = self._total_lines
        return env

    def simulate_step(self, action):
        env_copy = self._fast_copy()
        return env_copy.step(action)

    # -------------------------------------------------------------------------
    # step
    # -------------------------------------------------------------------------
    def step(self, action):
        """
        action = (rotation, x, y, path)   ← new format
                 (rotation, x)             ← legacy format (still supported)
        Returns: (next_state, reward, done)
        """
        shape, piece_id = self.current_shape
        shape = copy.deepcopy(shape)

        if len(action) == 2:
            rotation, x = action
            for _ in range(rotation):
                shape = rotate(shape)
            y = 0
            while self.board.is_valid_position(shape, x, y + 1):
                y += 1
        else:
            rotation, x, y, _path = action
            for _ in range(rotation):
                shape = rotate(shape)

        # Guard: invalid placement → game over
        if not self.board.is_valid_position(shape, x, y):
            return self.get_state_features(), -50.0, True

        # Lock piece (store colour so renderer can use it)
        color_id = piece_id + 1
        self.board.place_piece(shape, x, y, color_id=color_id)

        # Clear lines
        lines_cleared   = self.board.clear_lines()
        self._total_lines += lines_cleared

        # ── Reward shaping ─────────────────────────────────────────────
        heights   = self.board.column_heights()
        agg_h     = sum(heights)
        holes     = self.board.count_holes()
        bump      = self.board.bumpiness(heights)
        wells     = self.board.count_wells()
        row_trans = self.board.row_transitions()
        col_trans = self.board.col_transitions()

        # Dellacherie weights (proven strong for Tetris AI)
        reward = (
              lines_cleared * 100   # +100 per cleared line
            - holes        *  35    # −35  per hole
            - bump         *   3    # −3   per bumpiness unit
            - agg_h        *   0.5  # −0.5 per aggregate height unit
            - wells        *   5    # −5   per well depth unit
            - row_trans    *   3    # −3   per row transition
            - col_trans    *   3    # −3   per col transition
        )

        # Big bonus for Tetris (4 lines)
        if lines_cleared == 4:
            reward += 400
        elif lines_cleared == 3:
            reward += 100
        elif lines_cleared == 2:
            reward += 30

        # ── Advance piece queue ────────────────────────────────────────
        self.current_shape = self.next_shape
        self.next_shape    = get_random_piece()

        # Check game over on new piece spawn
        if not self.board.is_valid_position(self.current_shape[0], 3, 0):
            return self.get_state_features(), reward - 100.0, True

        return self.get_state_features(), reward, False

    # -------------------------------------------------------------------------
    # BFS over (x, y, rotation) reachable placements
    # -------------------------------------------------------------------------
    def get_valid_actions(self):
        shape_0 = self.current_shape[0]

        # Pre-build all rotation states
        rotated_shapes = [shape_0]
        seen_r = {tuple(tuple(r) for r in shape_0)}
        for _ in range(3):
            nxt = rotate(rotated_shapes[-1])
            key = tuple(tuple(r) for r in nxt)
            rotated_shapes.append(nxt)  # keep 4 entries; duplicates handled by visited set
            seen_r.add(key)

        # BFS
        start = (3, 0, 0)
        if not self.board.is_valid_position(rotated_shapes[0], 3, 0):
            return []

        queue   = [start]
        visited = {start}
        paths   = {start: [(0, 3, 0)]}
        valid_placements = {}           # key=(rot,x,y) → path

        while queue:
            cx, cy, cr = queue.pop(0)

            # Terminal: piece cannot drop further
            if not self.board.is_valid_position(rotated_shapes[cr], cx, cy + 1):
                key = (cr, cx, cy)
                p   = paths[(cx, cy, cr)]
                if key not in valid_placements or len(p) < len(valid_placements[key][3]):
                    valid_placements[key] = (cr, cx, cy, p)

            # Expand neighbours
            for nx, ny, nr in [
                (cx - 1, cy,     cr),           # move left
                (cx + 1, cy,     cr),           # move right
                (cx,     cy + 1, cr),           # drop one row
                (cx,     cy,     (cr + 1) % 4), # rotate CW
                (cx,     cy,     (cr - 1) % 4), # rotate CCW
            ]:
                if (nx, ny, nr) not in visited:
                    if self.board.is_valid_position(rotated_shapes[nr], nx, ny):
                        visited.add((nx, ny, nr))
                        queue.append((nx, ny, nr))
                        paths[(nx, ny, nr)] = paths[(cx, cy, cr)] + [(nr, nx, ny)]

        return list(valid_placements.values())

    # -------------------------------------------------------------------------
    # State feature vector
    # -------------------------------------------------------------------------
    def get_state_features(self):
        heights   = self.board.column_heights()
        agg_h     = sum(heights)
        holes     = self.board.count_holes()
        bump      = self.board.bumpiness(heights)
        wells     = self.board.count_wells()
        row_trans = self.board.row_transitions()
        col_trans = self.board.col_transitions()

        cur_id  = self.current_shape[1]
        nxt_id  = self.next_shape[1]
        cur_oh  = [1 if i == cur_id  else 0 for i in range(NUM_PIECES)]
        nxt_oh  = [1 if i == nxt_id  else 0 for i in range(NUM_PIECES)]

        features = (
            heights
            + [agg_h, holes, bump, wells, row_trans, col_trans]
            + cur_oh
            + nxt_oh
        )
        assert len(features) == STATE_DIM, f"STATE_DIM mismatch: {len(features)} vs {STATE_DIM}"
        return features