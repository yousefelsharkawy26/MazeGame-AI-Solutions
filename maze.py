import pygame
import random
import time
from pygame.locals import *

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 40
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Game setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        self.start_pos = (1, 1)
        self.end_pos = (width - 2, height - 2)
        self.generate_maze()
        self.player_pos = list(self.start_pos)
        
    def generate_maze(self):
        # Start with all walls
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = 1
        
        # Recursive backtracking algorithm
        stack = []
        x, y = self.start_pos
        self.grid[y][x] = 0
        stack.append((x, y))
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            
            # Find all unvisited neighbors
            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Remove wall between current cell and chosen neighbor
                self.grid[(y + ny) // 2][(x + nx) // 2] = 0
                self.grid[ny][nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Ensure start and end are clear
        self.grid[self.start_pos[1]][self.start_pos[0]] = 0
        self.grid[self.end_pos[1]][self.end_pos[0]] = 0
        
    def draw(self, surface):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[y][x] == 1:  # Wall
                    pygame.draw.rect(surface, BLACK, rect)
                else:  # Path
                    pygame.draw.rect(surface, WHITE, rect)
        
        # Draw start and end
        start_rect = pygame.Rect(self.start_pos[0] * CELL_SIZE, self.start_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        end_rect = pygame.Rect(self.end_pos[0] * CELL_SIZE, self.end_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, GREEN, start_rect)
        pygame.draw.rect(surface, RED, end_rect)
        
        # Draw player
        player_rect = pygame.Rect(self.player_pos[0] * CELL_SIZE, self.player_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, BLUE, player_rect)
    
    def move_player(self, dx, dy):
        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy
        
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            if self.grid[new_y][new_x] == 0:  # Only move if not a wall
                self.player_pos[0] = new_x
                self.player_pos[1] = new_y
                return True
        return False
    
    def check_win(self):
        return self.player_pos[0] == self.end_pos[0] and self.player_pos[1] == self.end_pos[1]

def main():
    maze_width = SCREEN_WIDTH // CELL_SIZE
    maze_height = SCREEN_HEIGHT // CELL_SIZE
    maze = Maze(maze_width, maze_height)
    
    running = True
    game_over = False
    start_time = time.time()
    elapsed_time = 0
    
    # Movement variables
    move_direction = [0, 0]  # [x, y]
    last_move_time = 0
    move_delay = 0.1  # seconds between moves when holding a key
    
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and not game_over:
                if event.key == K_UP:
                    move_direction = [0, -1]
                elif event.key == K_DOWN:
                    move_direction = [0, 1]
                elif event.key == K_LEFT:
                    move_direction = [-1, 0]
                elif event.key == K_RIGHT:
                    move_direction = [1, 0]
                elif event.key == K_r:  # Reset game
                    maze = Maze(maze_width, maze_height)
                    start_time = time.time()
                    game_over = False
                    move_direction = [0, 0]
            elif event.type == KEYUP and not game_over:
                # Stop movement when key is released
                if (event.key == K_UP and move_direction == [0, -1]) or \
                   (event.key == K_DOWN and move_direction == [0, 1]) or \
                   (event.key == K_LEFT and move_direction == [-1, 0]) or \
                   (event.key == K_RIGHT and move_direction == [1, 0]):
                    move_direction = [0, 0]
        
        # Continuous movement when key is held
        if not game_over and move_direction != [0, 0] and current_time - last_move_time > move_delay:
            maze.move_player(move_direction[0], move_direction[1])
            last_move_time = current_time
        
        # Update game state
        if not game_over:
            if maze.check_win():
                game_over = True
                elapsed_time = time.time() - start_time
            else:
                elapsed_time = time.time() - start_time
        
        # Draw everything
        screen.fill(GRAY)
        maze.draw(screen)
        
        # Draw time
        time_text = font.render(f"Time: {elapsed_time:.2f}s", True, BLACK)
        screen.blit(time_text, (10, 10))
        
        if game_over:
            win_text = font.render(f"You Win! Time: {elapsed_time:.2f}s (Press R to restart)", True, BLACK)
            text_rect = win_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            pygame.draw.rect(screen, WHITE, text_rect.inflate(20, 20))
            pygame.draw.rect(screen, BLACK, text_rect.inflate(20, 20), 2)
            screen.blit(win_text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()