import random
import time
import collections
import heapq

MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class MazeSolver:
    def __init__(self, maze, start_pos, end_pos):
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.reset()
    
    def reset(self):
        self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.current_pos = self.start_pos
        self.parent = {self.start_pos: None}
        self.solution_path = []
        self.solving = False
        self.solution_found = False
        self.steps = 0
        self.start_time = None
        self.end_time = None
        self.visited[self.start_pos[1]][self.start_pos[0]] = True
    
    def heuristic(self, pos):
        ex, ey = self.end_pos
        px, py = pos
        return abs(px - ex) + abs(py - ey)
    
    def start_solving(self):
        self.solving = True
        self.start_time = time.time()
        self.end_time = None
    
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

class DFSSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.stack = [self.start_pos]
    
    def solve_step(self):
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
                not self.visited[ny][nx] and self.maze[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                self.stack.append((nx, ny))
                
        return True

class BFSSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.queue = collections.deque([self.start_pos])
    
    def solve_step(self):
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
                not self.visited[ny][nx] and self.maze[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                self.queue.append((nx, ny))
                
        return True

class UCSSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.priority_queue = [(0, self.start_pos)]
    
    def solve_step(self):
        if not self.priority_queue:
            self.solving = False
            return False
            
        self.steps += 1
        cost, self.current_pos = heapq.heappop(self.priority_queue)
        
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
                not self.visited[ny][nx] and self.maze[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                heapq.heappush(self.priority_queue, (cost + 1, (nx, ny)))
                
        return True

class AStarSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.priority_queue = [(self.heuristic(self.start_pos), self.start_pos)]
    
    def solve_step(self):
        if not self.priority_queue:
            self.solving = False
            return False
            
        self.steps += 1
        _, self.current_pos = heapq.heappop(self.priority_queue)
        
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
                not self.visited[ny][nx] and self.maze[ny][nx] == 0):
                self.visited[ny][nx] = True
                self.parent[(nx, ny)] = (x, y)
                heapq.heappush(self.priority_queue, (self.heuristic((nx, ny)), (nx, ny)))
                
        return True

class GeneticSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.population = []
        self.population_size = 200
        self.chromosome_length = (end_pos[0] + 2) * (end_pos[1] + 2)
        self.generation = 1
        self.best_fitness = -1
        self.best_chromosome = None
        self.finished = False
    
    def random_chromosome(self):
        return [random.randint(0, 3) for _ in range(self.chromosome_length)]
    
    def decode_chromosome(self, chromosome):
        path = [self.start_pos]
        x, y = self.start_pos
        for gene in chromosome:
            dx, dy = MOVES[gene]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 0:
                x, y = nx, ny
                path.append((x, y))
                if (x, y) == self.end_pos:
                    break
        return path
    
    def start_solving(self):
        super().start_solving()
        self.population = [self.random_chromosome() for _ in range(self.population_size)]
        self.generation = 0
        self.best_fitness = -1
        self.best_chromosome = None
        self.finished = False
    
    def solve_step(self):
        if self.finished:
            return False

        # Evaluate current population
        fitnesses = []
        best_path = []
        for chromo in self.population:
            path = self.decode_chromosome(chromo)
            last_pos = path[-1]
            distance = abs(last_pos[0] - self.end_pos[0]) + abs(last_pos[1] - self.end_pos[1])
            
            # Fitness calculation
            if last_pos == self.end_pos:
                fitness = 1000 + (self.chromosome_length - len(path))
            else:
                fitness = (1 / (distance + 1)) * 500 + len(path) * 0.1
            
            fitnesses.append(fitness)
            
            # Track best path
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_chromosome = chromo
                best_path = path
                self.current_pos = path[-1]

        # Check for solution
        if best_path and best_path[-1] == self.end_pos:
            self.solution_path = best_path
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.finished = True
            return True

        # Create next generation
        new_population = []
        
        # Elitism: keep the best individual
        if self.best_chromosome:
            new_population.append(self.best_chromosome)
        
        while len(new_population) < self.population_size:
            # Tournament selection (size 3)
            candidates = random.sample(list(zip(self.population, fitnesses)), 3)
            parent1 = max(candidates, key=lambda x: x[1])[0]
            candidates = random.sample(list(zip(self.population, fitnesses)), 3)
            parent2 = max(candidates, key=lambda x: x[1])[0]
            
            # Crossover (80% chance)
            if random.random() < 0.8:
                point = random.randint(1, self.chromosome_length - 1)
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
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Update visualization with best path
        if self.best_chromosome:
            self.solution_path = self.decode_chromosome(self.best_chromosome)
        
        # Stop if taking too long
        if self.generation > 1000:
            self.solving = False
            self.end_time = time.time()
            self.finished = True
        
        return True

class IDSSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.depth_limit = 1
        self.current_depth = 0
        self.stack = [(self.start_pos, 0)]  # (position, current_depth)
        self.found_in_current_depth = False
    
    def reset(self):
        super().reset()
        self.depth_limit = 1
        self.current_depth = 0
        self.stack = [(self.start_pos, 0)]
        self.found_in_current_depth = False
    
    def start_solving(self):
        super().start_solving()
        self.depth_limit = 1
        self.current_depth = 0
        self.stack = [(self.start_pos, 0)]
        self.found_in_current_depth = False
    
    def solve_step(self):
        if not self.stack:
            # If we've exhausted this depth limit and didn't find the solution
            if not self.found_in_current_depth:
                # Increase depth limit and start over
                self.depth_limit += 1
                self.stack = [(self.start_pos, 0)]
                self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]
                self.visited[self.start_pos[1]][self.start_pos[0]] = True
                self.parent = {self.start_pos: None}
                self.current_depth = 0
                return True
            else:
                self.solving = False
                return False
        
        self.steps += 1
        self.current_pos, self.current_depth = self.stack.pop()
        
        if self.current_pos == self.end_pos:
            self.solution_found = True
            self.solving = False
            self.end_time = time.time()
            self.reconstruct_path()
            self.found_in_current_depth = True
            return True
            
        x, y = self.current_pos
        
        # Only explore deeper if we haven't reached the depth limit
        if self.current_depth < self.depth_limit:
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    not self.visited[ny][nx] and self.maze[ny][nx] == 0):
                    self.visited[ny][nx] = True
                    self.parent[(nx, ny)] = (x, y)
                    self.stack.append(((nx, ny), self.current_depth + 1))
        
        return True
    
class BSSolver(MazeSolver):
    def __init__(self, maze, start_pos, end_pos):
        super().__init__(maze, start_pos, end_pos)
        self.reset()
        
    def reset(self):
        super().reset()
        # Initialize forward search
        self.forward_queue = collections.deque([self.start_pos])
        self.forward_visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.forward_visited[self.start_pos[1]][self.start_pos[0]] = True
        self.forward_parent = {self.start_pos: None}
        
        # Initialize backward search
        self.backward_queue = collections.deque([self.end_pos])
        self.backward_visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.backward_visited[self.end_pos[1]][self.end_pos[0]] = True
        self.backward_parent = {self.end_pos: None}
        
        self.intersection = None
        self.current_pos = self.start_pos  # For visualization
        
    def solve_step(self):
        if not self.forward_queue or not self.backward_queue:
            self.solving = False
            return False
            
        self.steps += 1
        
        # Alternate between forward and backward steps
        if len(self.forward_queue) <= len(self.backward_queue):
            # Process forward search
            if self.forward_queue:
                current = self.forward_queue.popleft()
                self.current_pos = current  # Update visualization
                
                # Check if current node is visited by backward search
                if self.backward_visited[current[1]][current[0]]:
                    self.intersection = current
                    self.solution_found = True
                    self.solving = False
                    self.reconstruct_path()
                    return True
                
                # Explore neighbors
                x, y = current
                for dx, dy in MOVES:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and 
                        not self.forward_visited[ny][nx] and self.maze[ny][nx] == 0):
                        self.forward_visited[ny][nx] = True
                        self.forward_parent[(nx, ny)] = (x, y)
                        self.forward_queue.append((nx, ny))
        else:
            # Process backward search
            if self.backward_queue:
                current = self.backward_queue.popleft()
                
                # Check if current node is visited by forward search
                if self.forward_visited[current[1]][current[0]]:
                    self.intersection = current
                    self.solution_found = True
                    self.solving = False
                    self.reconstruct_path()
                    return True
                
                # Explore neighbors
                x, y = current
                for dx, dy in MOVES:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and 
                        not self.backward_visited[ny][nx] and self.maze[ny][nx] == 0):
                        self.backward_visited[ny][nx] = True
                        self.backward_parent[(nx, ny)] = (x, y)
                        self.backward_queue.append((nx, ny))
        
        return True
    
    def reconstruct_path(self):
        # Reconstruct path from start to intersection
        path = []
        node = self.intersection
        while node is not None:
            path.append(node)
            node = self.forward_parent.get(node)
        path.reverse()
        
        # Reconstruct path from intersection to end (excluding intersection)
        node = self.backward_parent.get(self.intersection)
        while node is not None:
            path.append(node)
            node = self.backward_parent.get(node)
            
        self.solution_path = path

