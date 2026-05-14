import pygame

CELL_SIZE = 30
COLUMNS = 10
ROWS = 20
BOARD_WIDTH = CELL_SIZE * COLUMNS
WIDTH = BOARD_WIDTH + 180
HEIGHT = CELL_SIZE * ROWS

BACKGROUND = (0, 0, 0)
BOARD_BG = (0, 0, 0)
PANEL_BG = (0, 0, 0)
GRID_COLOR = (0, 255, 0)
BORDER_COLOR = (255, 0, 0)
BLOCK_COLOR = (255, 0, 0)
PIECE_COLOR = (0, 255, 0)
TEXT_COLOR = (235, 235, 235)
GAME_OVER_COLOR = (220, 60, 60)

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tetris RL")
        self.title_font = pygame.font.SysFont(None, 28, bold=True)
        self.large_font = pygame.font.SysFont(None, 40, bold=True)
        self.small_font = pygame.font.SysFont(None, 20)

    def draw(self, board, shape, offset_x, offset_y, score=0, lines=0, level=1, next_piece=None, game_over=False):
        self.screen.fill(BACKGROUND)

        pygame.draw.rect(self.screen, BOARD_BG, (0, 0, BOARD_WIDTH, HEIGHT))
        pygame.draw.rect(self.screen, PANEL_BG, (BOARD_WIDTH, 0, WIDTH - BOARD_WIDTH, HEIGHT))

        for y in range(ROWS + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y * CELL_SIZE), (BOARD_WIDTH, y * CELL_SIZE), 1)
        for x in range(COLUMNS + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (x * CELL_SIZE, 0), (x * CELL_SIZE, HEIGHT), 1)

        for y in range(ROWS):
            for x in range(COLUMNS):
                if board.grid[y][x]:
                    pygame.draw.rect(
                        self.screen,
                        BLOCK_COLOR,
                        (x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                    )

        if shape is not None:
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(
                            self.screen,
                            PIECE_COLOR,
                            ((x + offset_x) * CELL_SIZE + 1,
                             (y + offset_y) * CELL_SIZE + 1,
                             CELL_SIZE - 2,
                             CELL_SIZE - 2)
                        )

        pygame.draw.rect(self.screen, BORDER_COLOR, (0, 0, BOARD_WIDTH, HEIGHT), 4)

        side_x = BOARD_WIDTH + 20
        self.screen.blit(self.title_font.render("TETRIS RL", True, TEXT_COLOR), (side_x, 20))
        self.screen.blit(self.small_font.render(f"SCORE", True, TEXT_COLOR), (side_x, 70))
        self.screen.blit(self.large_font.render(str(int(score)), True, TEXT_COLOR), (side_x, 100))

        self.screen.blit(self.small_font.render(f"LINES", True, TEXT_COLOR), (side_x, 170))
        self.screen.blit(self.large_font.render(str(lines), True, TEXT_COLOR), (side_x, 200))

        self.screen.blit(self.small_font.render(f"LEVEL", True, TEXT_COLOR), (side_x, 270))
        self.screen.blit(self.large_font.render(str(level), True, TEXT_COLOR), (side_x, 300))

        preview_label = self.small_font.render("NEXT", True, TEXT_COLOR)
        self.screen.blit(preview_label, (side_x, 360))

        if next_piece is not None:
            preview_size = 18
            preview_x = side_x
            preview_y = 390
            for y, row in enumerate(next_piece):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(
                            self.screen,
                            PIECE_COLOR,
                            (preview_x + x * preview_size,
                             preview_y + y * preview_size,
                             preview_size - 2,
                             preview_size - 2)
                        )

        hint = self.small_font.render("Press CLOSE to exit", True, TEXT_COLOR)
        self.screen.blit(hint, (side_x, HEIGHT - 40))

        if game_over:
            overlay = pygame.Surface((BOARD_WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            game_text = self.large_font.render("GAME OVER", True, GAME_OVER_COLOR)
            game_rect = game_text.get_rect(center=(BOARD_WIDTH // 2, HEIGHT // 2))
            self.screen.blit(game_text, game_rect)

        pygame.display.flip()