"""
Optimization algorithms for circuit board component placement.
"""
import numpy as np
import random
import copy
import time
from typing import List, Tuple, Dict, Optional, Callable

def optimize_placement(board: np.ndarray, method: str = "simulated_annealing", 
                      max_iterations: int = 1000) -> np.ndarray:
    """
    Optimize component placement on a circuit board.
    
    Args:
        board: Input board as a numpy array (height, width, features)
        method: Optimization method ('simulated_annealing' or 'genetic_algorithm')
        max_iterations: Maximum number of iterations
        
    Returns:
        Optimized board as a numpy array
    """
    if method == "simulated_annealing":
        return simulated_annealing(board, max_iterations)
    elif method == "genetic_algorithm":
        return genetic_algorithm(board, max_iterations)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

def calculate_score(board: np.ndarray) -> float:
    """
    Calculate a score for the board layout.
    Lower scores are better.
    
    Args:
        board: Board as a numpy array (height, width, features)
        
    Returns:
        Layout score
    """
    # Get board dimensions
    height, width = board.shape[0], board.shape[1]
    
    # Initialize score
    score = 0
    
    # Penalty for components too close to the edge
    edge_penalty = 0
    for y in range(height):
        for x in range(width):
            if board[y, x, 0] > 0:  # If there's a component
                # Check if it's on the edge
                if x == 0 or x == width-1 or y == 0 or y == height-1:
                    edge_penalty += 1
    
    # Penalty for high power components close to each other
    power_penalty = 0
    for y in range(height):
        for x in range(width):
            if board[y, x, 0] > 0:  # If there's a component
                power = board[y, x, 3]
                
                # Check surrounding cells
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= y+dy < height and 0 <= x+dx < width and (dx != 0 or dy != 0):
                            if board[y+dy, x+dx, 0] > 0:  # If there's another component
                                neighbor_power = board[y+dy, x+dx, 3]
                                power_penalty += power * neighbor_power
    
    # Calculate final score (lower is better)
    score = edge_penalty * 10 + power_penalty * 5
    
    return score

def get_components(board: np.ndarray) -> List[Dict]:
    """
    Extract components from a board.
    
    Args:
        board: Board as a numpy array (height, width, features)
        
    Returns:
        List of components with their properties
    """
    height, width = board.shape[0], board.shape[1]
    components = []
    visited = set()
    
    for y in range(height):
        for x in range(width):
            if board[y, x, 0] > 0 and (y, x) not in visited:
                # Found a component
                comp_type = int(board[y, x, 0])
                comp_width = int(board[y, x, 1])
                comp_height = int(board[y, x, 2])
                power = board[y, x, 3]
                
                # Mark all cells of this component as visited
                for dy in range(comp_height):
                    for dx in range(comp_width):
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            visited.add((y+dy, x+dx))
                
                # Add component to list
                components.append({
                    "type": comp_type,
                    "width": comp_width,
                    "height": comp_height,
                    "power": power,
                    "position": (x, y)
                })
    
    return components

def place_components(board_shape: Tuple[int, int], components: List[Dict]) -> np.ndarray:
    """
    Place components on a new board.
    
    Args:
        board_shape: (height, width) of the board
        components: List of components with their properties
        
    Returns:
        New board with placed components
    """
    height, width = board_shape
    board = np.zeros((height, width, 4))
    
    for comp in components:
        x, y = comp["position"]
        comp_type = comp["type"]
        comp_width = comp["width"]
        comp_height = comp["height"]
        power = comp["power"]
        
        # Place component
        for dy in range(comp_height):
            for dx in range(comp_width):
                if 0 <= y+dy < height and 0 <= x+dx < width:
                    board[y+dy, x+dx, 0] = comp_type
                    board[y+dy, x+dx, 1] = comp_width
                    board[y+dy, x+dx, 2] = comp_height
                    board[y+dy, x+dx, 3] = power
    
    return board

def is_valid_placement(board_shape: Tuple[int, int], components: List[Dict]) -> bool:
    """
    Check if component placement is valid (no overlaps, within bounds).
    
    Args:
        board_shape: (height, width) of the board
        components: List of components with their properties
        
    Returns:
        True if placement is valid, False otherwise
    """
    height, width = board_shape
    occupied = np.zeros((height, width), dtype=bool)
    
    for comp in components:
        x, y = comp["position"]
        comp_width = comp["width"]
        comp_height = comp["height"]
        
        # Check if component is within bounds
        if x < 0 or x + comp_width > width or y < 0 or y + comp_height > height:
            return False
        
        # Check for overlaps
        for dy in range(comp_height):
            for dx in range(comp_width):
                if occupied[y+dy, x+dx]:
                    return False
                occupied[y+dy, x+dx] = True
    
    return True

def simulated_annealing(board: np.ndarray, max_iterations: int = 1000,
                       initial_temp: float = 100.0, cooling_rate: float = 0.95) -> np.ndarray:
    """
    Optimize component placement using simulated annealing.
    
    Args:
        board: Input board as a numpy array
        max_iterations: Maximum number of iterations
        initial_temp: Initial temperature
        cooling_rate: Cooling rate
        
    Returns:
        Optimized board
    """
    # Get board dimensions
    height, width = board.shape[0], board.shape[1]
    
    # Extract components
    components = get_components(board)
    
    # Calculate initial score
    current_score = calculate_score(board)
    best_score = current_score
    best_components = copy.deepcopy(components)
    
    # Initialize temperature
    temp = initial_temp
    
    # Run simulated annealing
    for iteration in range(max_iterations):
        # Make a copy of components
        new_components = copy.deepcopy(components)
        
        # Choose a random component
        if not new_components:
            continue
        comp_idx = random.randint(0, len(new_components) - 1)
        comp = new_components[comp_idx]
        
        # Choose a new random position
        old_pos = comp["position"]
        new_x = random.randint(0, width - comp["width"])
        new_y = random.randint(0, height - comp["height"])
        comp["position"] = (new_x, new_y)
        
        # Check if new placement is valid
        if is_valid_placement((height, width), new_components):
            # Create new board
            new_board = place_components((height, width), new_components)
            
            # Calculate new score
            new_score = calculate_score(new_board)
            
            # Decide whether to accept the new solution
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / temp):
                # Accept new solution
                components = new_components
                current_score = new_score
                
                # Update best solution
                if new_score < best_score:
                    best_score = new_score
                    best_components = copy.deepcopy(components)
            else:
                # Revert position
                comp["position"] = old_pos
        
        # Cool down
        temp *= cooling_rate
    
    # Create final board with best components
    best_board = place_components((height, width), best_components)
    
    return best_board

def genetic_algorithm(board: np.ndarray, max_iterations: int = 50,
                     population_size: int = 20, mutation_rate: float = 0.2) -> np.ndarray:
    """
    Optimize component placement using a genetic algorithm.
    
    Args:
        board: Input board as a numpy array
        max_iterations: Maximum number of generations
        population_size: Size of the population
        mutation_rate: Probability of mutation
        
    Returns:
        Optimized board
    """
    # Get board dimensions
    height, width = board.shape[0], board.shape[1]
    
    # Extract components
    components = get_components(board)
    
    # Create initial population
    population = []
    for _ in range(population_size):
        # Create a copy of components
        new_components = copy.deepcopy(components)
        
        # Randomize positions
        for comp in new_components:
            valid_placement = False
            attempts = 0
            
            while not valid_placement and attempts < 100:
                # Choose a new random position
                new_x = random.randint(0, width - comp["width"])
                new_y = random.randint(0, height - comp["height"])
                comp["position"] = (new_x, new_y)
                
                # Check if placement is valid
                valid_placement = is_valid_placement((height, width), new_components)
                attempts += 1
        
        # Add to population
        population.append(new_components)
    
    # Run genetic algorithm
    for generation in range(max_iterations):
        # Calculate fitness for each individual
        fitness = []
        for individual in population:
            # Create board
            individual_board = place_components((height, width), individual)
            
            # Calculate score (lower is better)
            score = calculate_score(individual_board)
            
            # Convert to fitness (higher is better)
            fitness.append(1.0 / (score + 1.0))
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = np.argmax(fitness)
        new_population.append(copy.deepcopy(population[best_idx]))
        
        # Create rest of population through selection, crossover, and mutation
        while len(new_population) < population_size:
            # Selection (tournament selection)
            parent1_idx = random.randint(0, population_size - 1)
            parent2_idx = random.randint(0, population_size - 1)
            
            if fitness[parent1_idx] > fitness[parent2_idx]:
                parent1 = population[parent1_idx]
            else:
                parent1 = population[parent2_idx]
            
            parent2_idx = random.randint(0, population_size - 1)
            parent3_idx = random.randint(0, population_size - 1)
            
            if fitness[parent2_idx] > fitness[parent3_idx]:
                parent2 = population[parent2_idx]
            else:
                parent2 = population[parent3_idx]
            
            # Crossover
            child = []
            for i in range(len(components)):
                if random.random() < 0.5:
                    child.append(copy.deepcopy(parent1[i]))
                else:
                    child.append(copy.deepcopy(parent2[i]))
            
            # Mutation
            for comp in child:
                if random.random() < mutation_rate:
                    # Choose a new random position
                    new_x = random.randint(0, width - comp["width"])
                    new_y = random.randint(0, height - comp["height"])
                    comp["position"] = (new_x, new_y)
            
            # Check if placement is valid
            if is_valid_placement((height, width), child):
                new_population.append(child)
        
        # Replace population
        population = new_population
    
    # Get best individual
    fitness = []
    for individual in population:
        # Create board
        individual_board = place_components((height, width), individual)
        
        # Calculate score
        score = calculate_score(individual_board)
        
        # Convert to fitness
        fitness.append(1.0 / (score + 1.0))
    
    best_idx = np.argmax(fitness)
    best_components = population[best_idx]
    
    # Create final board with best components
    best_board = place_components((height, width), best_components)
    
    return best_board 