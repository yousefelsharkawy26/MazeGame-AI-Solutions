import pygame
import random
from solver import DFSSolver, BFSSolver, UCSSolver, AStarSolver, GeneticSolver, IDSSolver, BSSolver
from pygame.locals import (
    QUIT, MOUSEBUTTONDOWN
    ,KEYDOWN, K_r, K_SPACE
)

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
MAZE_WIDTH = 900
CELL_SIZE = 40
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 180, 0)
SIDEBAR_COLOR = (220, 220, 220)
DFS_COLOR = (100, 100, 255)
IDS_COLOR = (120, 150, 255)
BFS_COLOR = (255, 100, 100)
UCS_COLOR = (100, 255, 100)
GREEDY_COLOR = (255, 255, 100)
GA_COLOR = (150, 50, 200)
BS_COLOR = (150, 200, 255) 
BUTTON_HOVER_COLOR = (180, 180, 180)

# Moves (up, right, down, left)
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.is_hovered = False
        self.font = pygame.font.SysFont('Arial', 20)
        
    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
        
    def is_clicked(self, pos, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

class MazeGenerator:
    @staticmethod
    def generate_maze(width, height, start_pos=(1, 1), end_pos=None):
        if end_pos is None:
            end_pos = (width - 2, height - 2)
        
        grid = [[1 for _ in range(width)] for _ in range(height)]
        
        # Recursive backtracking algorithm
        stack = []
        x, y = start_pos
        grid[y][x] = 0
        stack.append((x, y))
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            
            # Find all unvisited neighbors
            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Remove wall between current cell and chosen neighbor
                grid[(y + ny) // 2][(x + nx) // 2] = 0
                grid[ny][nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Ensure start and end are clear
        grid[start_pos[1]][start_pos[0]] = 0
        grid[end_pos[1]][end_pos[0]] = 0
        
        return grid, start_pos, end_pos

class MazeVisualizer:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.maze = None
        self.solver = None
        self.buttons = []
        self.font = pygame.font.SysFont('Arial', 24)
        self.button_font = pygame.font.SysFont('Arial', 20)
        
    def create_buttons(self):
        # Create buttons
        buttons = [
            Button(MAZE_WIDTH + 20, 50, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve DFS", DFS_COLOR),
            Button(MAZE_WIDTH + 20, 90, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve IDS", IDS_COLOR),
            Button(MAZE_WIDTH + 20, 130, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve BFS", BFS_COLOR),
            Button(MAZE_WIDTH + 20, 170, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve UCS", UCS_COLOR),
            Button(MAZE_WIDTH + 20, 210, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve A*", GREEDY_COLOR),
            Button(MAZE_WIDTH + 20, 250, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve GENETIC", GA_COLOR),
            Button(MAZE_WIDTH + 20, 170, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve BS", BS_COLOR),
            Button(MAZE_WIDTH + 20, 290, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Reset Maze", (200, 200, 200))
        ]
        return buttons
    
    def draw_maze(self, surface):
        if self.maze is None:
            return
            
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                self.cell_size, self.cell_size)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(surface, BLACK, rect)
                elif self.solver and self.solver.visited[y][x]:
                    pygame.draw.rect(surface, YELLOW, rect)
                else:
                    pygame.draw.rect(surface, WHITE, rect)
        
        # Draw start and end (they exist even without solver)
        pygame.draw.rect(surface, GREEN, 
                        pygame.Rect(1 * self.cell_size, 
                                    1 * self.cell_size, 
                                    self.cell_size, self.cell_size))
        pygame.draw.rect(surface, RED, 
                        pygame.Rect((self.width-2) * self.cell_size, 
                                   (self.height-2) * self.cell_size, 
                                    self.cell_size, self.cell_size))
        
        # Only draw solver-related elements if solver exists
        if self.solver:
            # Draw current position
            pygame.draw.rect(surface, BLUE, 
                            pygame.Rect(self.solver.current_pos[0]*self.cell_size, 
                                        self.solver.current_pos[1]*self.cell_size, 
                                        self.cell_size, self.cell_size))
            
            # Draw solution path
            if self.solver.solution_path:
                for pos in self.solver.solution_path:
                    if pos not in (self.solver.start_pos, self.solver.end_pos):
                        pygame.draw.rect(surface, DARK_GREEN, 
                                        pygame.Rect(pos[0]*self.cell_size, 
                                                    pos[1]*self.cell_size, 
                                                    self.cell_size, self.cell_size))
    
    def draw_sidebar(self, surface):
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)
            button.draw(surface)
        
        # Display info - only show solver stats if solver exists
        if self.solver:
            time_text = self.font.render(f"Time: {self.solver.get_elapsed_time():.2f}s", True, BLACK)
            steps_text = self.font.render(f"Steps: {self.solver.steps}", True, BLACK)
            
            status = "Ready"
            if self.solver.solution_found:
                status = "Solution Found"
            elif self.solver.solving:
                status = "Solving"
            status_text = self.font.render(f"Status: {status}", True, BLACK)
            
            surface.blit(time_text, (MAZE_WIDTH + 20, 410))
            surface.blit(steps_text, (MAZE_WIDTH + 20, 450))
            surface.blit(status_text, (MAZE_WIDTH + 20, 490))
            
            # Display GA info if applicable
            if isinstance(self.solver, GeneticSolver):
                ga_info = [
                    f"Generation: {self.solver.generation}",
                    f"Best Fitness: {self.solver.best_fitness:.1f}",
                    f"Current Pos: {self.solver.current_pos}"
                ]
                for i, text in enumerate(ga_info):
                    text_surface = self.font.render(text, True, BLACK)
                    surface.blit(text_surface, (MAZE_WIDTH + 20, 530 + i * 30))
        else:
            # Show basic status when no solver is active
            status_text = self.font.render("Status: Ready (select algorithm)", True, BLACK)
            surface.blit(status_text, (MAZE_WIDTH + 20, 410))

class MazeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Maze Solver - DFS/BFS/UCS/Greedy/Genetic")
        self.clock = pygame.time.Clock()
        
        self.visualizer = MazeVisualizer(MAZE_WIDTH // CELL_SIZE, SCREEN_HEIGHT // CELL_SIZE, CELL_SIZE)
        self.visualizer.buttons = self.visualizer.create_buttons()
        self.generate_new_maze()
    
    def generate_new_maze(self):
        width = MAZE_WIDTH // CELL_SIZE
        height = SCREEN_HEIGHT // CELL_SIZE
        self.maze, start_pos, end_pos = MazeGenerator.generate_maze(width, height)
        self.visualizer.maze = self.maze
        self.visualizer.solver = None
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE and self.visualizer.solver and self.visualizer.solver.solving:
                    self.visualizer.solver.solving = False
                elif event.key == K_r:
                    self.generate_new_maze()
            elif event.type == MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(self.visualizer.buttons):
                    if button.is_clicked(mouse_pos, event):
                        if i == 0:  # DFS
                            self.visualizer.solver = DFSSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 1:  # IDS
                            self.visualizer.solver = IDSSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 2:  # BFS
                            self.visualizer.solver = BFSSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 3:  # UCS
                            self.visualizer.solver = UCSSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 4:  # A*
                            self.visualizer.solver = AStarSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 5:  # GA
                            self.visualizer.solver = GeneticSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 6:  # GA
                            self.visualizer.solver = BSSolver(self.visualizer.maze, (1, 1), 
                                                            (len(self.visualizer.maze[0])-2, 
                                                            len(self.visualizer.maze)-2))
                        elif i == 7:  # Reset
                            self.generate_new_maze()
                            return True
                        
                        if i < 7:  # Any solver button (now 6 buttons before Reset)
                            self.visualizer.solver.start_solving()
        return True
    
    def update(self):
        if self.visualizer.solver and self.visualizer.solver.solving:
            if isinstance(self.visualizer.solver, GeneticSolver):
                # Run multiple GA steps per frame for better performance
                for _ in range(5):
                    if not self.visualizer.solver.solve_step():
                        break
            else:
                self.visualizer.solver.solve_step()
    
    def render(self):
        self.screen.fill(SIDEBAR_COLOR)
        #if self.visualizer.maze and self.visualizer.solver:
        self.visualizer.draw_maze(self.screen)
        self.visualizer.draw_sidebar(self.screen)
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()

