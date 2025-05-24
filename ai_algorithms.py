import pygame
import random
import time
import collections
import heapq
from pygame.locals import *

# Initialize pygame
pygame.init()

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
BFS_COLOR = (255, 100, 100)
UCS_COLOR = (100, 255, 100)
GREEDY_COLOR = (255, 255, 100)
GA_COLOR = (150, 50, 200)
BUTTON_HOVER_COLOR = (180, 180, 180)

# Moves (up, right, down, left)
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Game setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze Solver - DFS/BFS/UCS/Greedy/Genetic")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)
button_font = pygame.font.SysFont('Arial', 20)

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.is_hovered = False
        
    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        text_surface = button_font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered
        
    def is_clicked(self, pos, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        self.start_pos = (1, 1)
        self.end_pos = (width - 2, height - 2)
        self.generate_maze()
        self.reset_solver()
        
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
    
    def reset_solver(self):
        self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.stack = [self.start_pos]
        self.queue = collections.deque([self.start_pos])
        self.ucs_queue = [(0, self.start_pos)]
        self.greedy_queue = [(self.heuristic(self.start_pos), self.start_pos)]
        self.current_pos = self.start_pos
        self.parent = {self.start_pos: None}
        self.solution_path = []
        self.solving = False
        self.solution_found = False
        self.steps = 0
        self.start_time = None
        self.end_time = None
        self.algorithm = None
        self.visited[self.start_pos[1]][self.start_pos[0]] = True
        
        # GA specific
        self.ga_population = []
        self.ga_population_size = 200  # Increased from 100
        self.ga_chromosome_length = (MAZE_WIDTH + SCREEN_HEIGHT) # Longer chromosomes
        self.ga_generation = 0
        self.ga_best_fitness = -1
        self.ga_best_chromosome = None
        self.ga_finished = False
        
    def heuristic(self, pos):
        ex, ey = self.end_pos
        px, py = pos
        return abs(px - ex) + abs(py - ey)
    
    def start_solving(self, algorithm):
        self.solving = True
        self.algorithm = algorithm
        self.start_time = time.time()
        self.end_time = None
        
        if algorithm == 'GA':
            self.ga_population = [self.random_chromosome() for _ in range(self.ga_population_size)]
            self.ga_generation = 0
            self.ga_best_fitness = -1
            self.ga_best_chromosome = None
            self.ga_finished = False
    
    def random_chromosome(self):
        return [random.randint(0, 3) for _ in range(self.ga_chromosome_length)]
    
    def decode_chromosome(self, chromosome):
        path = [self.start_pos]
        x, y = self.start_pos
        for gene in chromosome:
            dx, dy = MOVES[gene]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                x, y = nx, ny
                path.append((x, y))
                if (x, y) == self.end_pos:
                    break
        return path
    
    def genetic_solve(self):
        if self.ga_finished:
            return False

        # Evaluate current population
        fitnesses = []
        best_path = []
        for chromo in self.ga_population:
            path = self.decode_chromosome(chromo)
            last_pos = path[-1]
            distance = abs(last_pos[0] - self.end_pos[0]) + abs(last_pos[1] - self.end_pos[1])
            
            # Fitness calculation
            if last_pos == self.end_pos:
                fitness = 1000 + (self.ga_chromosome_length - len(path))
            else:
                fitness = (1 / (distance + 1)) * 500 + len(path) * 0.1
            
            fitnesses.append(fitness)
            
            # Track best path
            if fitness > self.ga_best_fitness:
                self.ga_best_fitness = fitness
                self.ga_best_chromosome = chromo
                best_path = path
                self.current_pos = path[-1]

        # Check for solution
        if best_path and best_path[-1] == self.end_pos:
            self.solution_path = best_path
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.ga_finished = True
            return True

        # Create next generation
        new_population = []
        
        # Elitism: keep the best individual
        if self.ga_best_chromosome:
            new_population.append(self.ga_best_chromosome)
        
        while len(new_population) < self.ga_population_size:
            # Tournament selection (size 3)
            candidates = random.sample(list(zip(self.ga_population, fitnesses)), 3)
            parent1 = max(candidates, key=lambda x: x[1])[0]
            candidates = random.sample(list(zip(self.ga_population, fitnesses)), 3)
            parent2 = max(candidates, key=lambda x: x[1])[0]
            
            # Crossover (80% chance)
            if random.random() < 0.8:
                point = random.randint(1, self.ga_chromosome_length - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            else:
                child1, child2 = parent1[:], parent2[:]
                
            # Mutation (5% per gene)
            for i in range(len(child1)):
                if random.random() < 0.05:
                    child1[i] = random.randint(0, 3)
            for i in range(len(child2)):
                if random.random() < 0.05:
                    child2[i] = random.randint(0, 3)
                    
            new_population.extend([child1, child2])
        
        self.ga_population = new_population[:self.ga_population_size]
        self.ga_generation += 1
        
        # Update visualization with best path
        if self.ga_best_chromosome:
            self.solution_path = self.decode_chromosome(self.ga_best_chromosome)
        
        # Stop if taking too long
        if self.ga_generation > 1000:
            self.solving = False
            self.end_time = time.time()
            self.ga_finished = True
        
        return True
    
    def draw(self, surface):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[y][x] == 1:
                    pygame.draw.rect(surface, BLACK, rect)
                elif self.visited[y][x]:
                    pygame.draw.rect(surface, YELLOW, rect)
                else:
                    pygame.draw.rect(surface, WHITE, rect)
        
        # Draw start and end
        pygame.draw.rect(surface, GREEN, 
                        pygame.Rect(self.start_pos[0]*CELL_SIZE, self.start_pos[1]*CELL_SIZE, 
                                   CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(surface, RED, 
                        pygame.Rect(self.end_pos[0]*CELL_SIZE, self.end_pos[1]*CELL_SIZE, 
                                   CELL_SIZE, CELL_SIZE))
        
        # Draw current position
        pygame.draw.rect(surface, BLUE, 
                        pygame.Rect(self.current_pos[0]*CELL_SIZE, self.current_pos[1]*CELL_SIZE, 
                                   CELL_SIZE, CELL_SIZE))
        
        # Draw solution path
        if self.solution_path:
            for pos in self.solution_path:
                if pos not in (self.start_pos, self.end_pos):
                    pygame.draw.rect(surface, DARK_GREEN, 
                                    pygame.Rect(pos[0]*CELL_SIZE, pos[1]*CELL_SIZE, 
                                               CELL_SIZE, CELL_SIZE))
    
    def dfs_solve(self):
        if not self.stack:
            self.solving = False
            return False
            
        self.steps += 1
        self.current_pos = self.stack.pop()
        
        if self.current_pos == self.end_pos:
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.reconstruct_path()
            return True
            
        x, y = self.current_pos
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                not self.visited[ny][nx] and self.grid[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                self.stack.append((nx, ny))
                
        return True
    
    def bfs_solve(self):
        if not self.queue:
            self.solving = False
            return False
            
        self.steps += 1
        self.current_pos = self.queue.popleft()
        
        if self.current_pos == self.end_pos:
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.reconstruct_path()
            return True
            
        x, y = self.current_pos
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                not self.visited[ny][nx] and self.grid[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                self.queue.append((nx, ny))
                
        return True
    
    def ucs_solve(self):
        if not self.ucs_queue:
            self.solving = False
            return False
            
        self.steps += 1
        cost, self.current_pos = heapq.heappop(self.ucs_queue)
        
        if self.current_pos == self.end_pos:
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.reconstruct_path()
            return True
            
        x, y = self.current_pos
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                not self.visited[ny][nx] and self.grid[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                heapq.heappush(self.ucs_queue, (cost + 1, (nx, ny)))
                
        return True
    
    def greedy_solve(self):
        if not self.greedy_queue:
            self.solving = False
            return False
            
        self.steps += 1
        _, self.current_pos = heapq.heappop(self.greedy_queue)
        
        if self.current_pos == self.end_pos:
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.reconstruct_path()
            return True
            
        x, y = self.current_pos
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                not self.visited[ny][nx] and self.grid[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                heapq.heappush(self.greedy_queue, (self.heuristic((nx, ny)), (nx, ny)))
                
        return True
    
    def solve_step(self):
        if self.algorithm == 'DFS':
            return self.dfs_solve()
        elif self.algorithm == 'BFS':
            return self.bfs_solve()
        elif self.algorithm == 'UCS':
            return self.ucs_solve()
        elif self.algorithm == 'Greedy':
            return self.greedy_solve()
        elif self.algorithm == 'GA':
            # Run multiple GA steps per frame for better performance
            for _ in range(5):
                if not self.genetic_solve():
                    return False
            return True
        return False
    
    def reconstruct_path(self):
        self.solution_path = []
        current = self.end_pos
        while current is not None:
            self.solution_path.append(current)
            current = self.parent.get(current)
        self.solution_path.reverse()
    
    def get_elapsed_time(self):
        if not self.start_time:
            return 0.0
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

def main():
    maze = Maze(MAZE_WIDTH // CELL_SIZE, SCREEN_HEIGHT // CELL_SIZE)
    
    # Create buttons
    dfs_button = Button(MAZE_WIDTH + 20, 50, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve DFS", DFS_COLOR)
    bfs_button = Button(MAZE_WIDTH + 20, 110, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve BFS", BFS_COLOR)
    ucs_button = Button(MAZE_WIDTH + 20, 170, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve UCS", UCS_COLOR)
    greedy_button = Button(MAZE_WIDTH + 20, 230, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve Greedy", GREEDY_COLOR)
    ga_button = Button(MAZE_WIDTH + 20, 290, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Solve GENETIC", GA_COLOR)
    reset_button = Button(MAZE_WIDTH + 20, 350, SCREEN_WIDTH-MAZE_WIDTH-40, 40, "Reset Maze", (200, 200, 200))
    
    running = True
    while running:
        screen.fill(SIDEBAR_COLOR)
        maze.draw(screen)
        
        mouse_pos = pygame.mouse.get_pos()
        
        # Draw buttons
        for button in [dfs_button, bfs_button, ucs_button, greedy_button, ga_button, reset_button]:
            button.check_hover(mouse_pos)
            button.draw(screen)
        
        # Display info
        time_text = font.render(f"Time: {maze.get_elapsed_time():.2f}s", True, BLACK)
        steps_text = font.render(f"Steps: {maze.steps}", True, BLACK)
        status = "Ready"
        if maze.solution_found:
            status = f"Solved by {maze.algorithm}"
        elif maze.solving:
            status = f"Solving by {maze.algorithm}"
        status_text = font.render(f"Status: {status}", True, BLACK)
        
        screen.blit(time_text, (MAZE_WIDTH + 20, 410))
        screen.blit(steps_text, (MAZE_WIDTH + 20, 450))
        screen.blit(status_text, (MAZE_WIDTH + 20, 490))
        
        # Display GA info if applicable
        if maze.algorithm == 'GA':
            ga_info = [
                f"Generation: {maze.ga_generation}",
                f"Best Fitness: {maze.ga_best_fitness:.1f}",
                f"Current Pos: {maze.current_pos}"
            ]
            for i, text in enumerate(ga_info):
                text_surface = font.render(text, True, BLACK)
                screen.blit(text_surface, (MAZE_WIDTH + 20, 530 + i * 30))
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE and maze.solving:
                    maze.solving = False
                elif event.key == K_r:
                    maze.generate_maze()
                    maze.reset_solver()
            elif event.type == MOUSEBUTTONDOWN:
                if dfs_button.is_clicked(mouse_pos, event):
                    maze.reset_solver()
                    maze.start_solving('DFS')
                elif bfs_button.is_clicked(mouse_pos, event):
                    maze.reset_solver()
                    maze.start_solving('BFS')
                elif ucs_button.is_clicked(mouse_pos, event):
                    maze.reset_solver()
                    maze.start_solving('UCS')
                elif greedy_button.is_clicked(mouse_pos, event):
                    maze.reset_solver()
                    maze.start_solving('Greedy')
                elif ga_button.is_clicked(mouse_pos, event):
                    maze.reset_solver()
                    maze.start_solving('GA')
                elif reset_button.is_clicked(mouse_pos, event):
                    maze.generate_maze()
                    maze.reset_solver()
        
        if maze.solving:
            maze.solve_step()
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()