"""
Component definitions for circuit board placement.
"""
import enum
import numpy as np
from typing import List, Tuple, Dict, Optional

class ComponentType(enum.Enum):
    """Enum for component types."""
    RESISTOR = 1
    CAPACITOR = 2
    INDUCTOR = 3
    TRANSISTOR = 4
    IC = 5

class Component:
    """Class representing a circuit board component."""
    
    def __init__(self, 
                id: int, 
                type: ComponentType, 
                width: int, 
                height: int, 
                power_rating: float,
                connections: List[int] = None,
                position: Tuple[int, int] = None):
        """
        Initialize a component.
        
        Args:
            id: Unique identifier
            type: Component type
            width: Width in grid cells
            height: Height in grid cells
            power_rating: Power rating in watts
            connections: List of connected component IDs
            position: (x, y) position on the board
        """
        self.id = id
        self.type = type
        self.width = width
        self.height = height
        self.power_rating = power_rating
        self.connections = connections or []
        self.position = position
    
    def __repr__(self):
        return f"Component(id={self.id}, type={self.type.name}, pos={self.position})"

class Board:
    """Class representing a circuit board."""
    
    def __init__(self, width: int, height: int, components: List[Component]):
        """
        Initialize a board.
        
        Args:
            width: Board width in grid cells
            height: Board height in grid cells
            components: List of components on the board
        """
        self.width = width
        self.height = height
        self.components = components
    
    def to_grid(self) -> np.ndarray:
        """
        Convert the board to a grid representation.
        
        Returns:
            A grid where each cell contains the component ID or 0 for empty cells
        """
        grid = np.zeros((self.height, self.width), dtype=int)
        
        for comp in self.components:
            if comp.position is not None:
                x, y = comp.position
                for dy in range(comp.height):
                    for dx in range(comp.width):
                        if 0 <= y + dy < self.height and 0 <= x + dx < self.width:
                            grid[y + dy, x + dx] = comp.id
        
        return grid
    
    def to_feature_grid(self) -> np.ndarray:
        """
        Convert the board to a feature grid representation.
        
        Returns:
            A grid with features for each cell:
                - Channel 0: Component type (0 for empty)
                - Channel 1: Component width
                - Channel 2: Component height
                - Channel 3: Power rating
        """
        grid = np.zeros((self.height, self.width, 4), dtype=float)
        
        for comp in self.components:
            if comp.position is not None:
                x, y = comp.position
                for dy in range(comp.height):
                    for dx in range(comp.width):
                        if 0 <= y + dy < self.height and 0 <= x + dx < self.width:
                            grid[y + dy, x + dx, 0] = comp.type.value
                            grid[y + dy, x + dx, 1] = comp.width
                            grid[y + dy, x + dx, 2] = comp.height
                            grid[y + dy, x + dx, 3] = comp.power_rating
        
        return grid
    
    def layout_score(self) -> float:
        """
        Calculate a score for the current layout.
        Lower scores are better.
        
        Returns:
            Layout score
        """
        # Convert to feature grid
        grid = self.to_feature_grid()
        
        # Initialize score
        score = 0
        
        # Penalty for components too close to the edge
        edge_penalty = 0
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x, 0] > 0:  # If there's a component
                    # Check if it's on the edge
                    if x == 0 or x == self.width-1 or y == 0 or y == self.height-1:
                        edge_penalty += 1
        
        # Penalty for high power components close to each other
        power_penalty = 0
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x, 0] > 0:  # If there's a component
                    power = grid[y, x, 3]
                    
                    # Check surrounding cells
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if 0 <= y+dy < self.height and 0 <= x+dx < self.width and (dx != 0 or dy != 0):
                                if grid[y+dy, x+dx, 0] > 0:  # If there's another component
                                    neighbor_power = grid[y+dy, x+dx, 3]
                                    power_penalty += power * neighbor_power
        
        # Connection length penalty
        connection_penalty = 0
        for comp1 in self.components:
            if comp1.position is not None:
                x1, y1 = comp1.position
                for conn_id in comp1.connections:
                    # Find the connected component
                    for comp2 in self.components:
                        if comp2.id == conn_id and comp2.position is not None:
                            x2, y2 = comp2.position
                            # Calculate Manhattan distance
                            distance = abs(x2 - x1) + abs(y2 - y1)
                            connection_penalty += distance
        
        # Calculate final score (lower is better)
        score = edge_penalty * 10 + power_penalty * 5 + connection_penalty * 2
        
        return score

def create_component_library() -> Dict[ComponentType, List[Dict]]:
    """
    Create a library of component templates.
    
    Returns:
        Dictionary mapping component types to lists of templates
    """
    library = {
        ComponentType.RESISTOR: [
            {"width": 1, "height": 2, "power_rating": 0.1},
            {"width": 2, "height": 1, "power_rating": 0.2},
        ],
        ComponentType.CAPACITOR: [
            {"width": 1, "height": 1, "power_rating": 0.1},
            {"width": 2, "height": 2, "power_rating": 0.3},
        ],
        ComponentType.INDUCTOR: [
            {"width": 2, "height": 2, "power_rating": 0.5},
            {"width": 3, "height": 1, "power_rating": 0.4},
        ],
        ComponentType.TRANSISTOR: [
            {"width": 1, "height": 1, "power_rating": 0.3},
            {"width": 2, "height": 1, "power_rating": 0.6},
        ],
        ComponentType.IC: [
            {"width": 3, "height": 3, "power_rating": 1.5},
            {"width": 2, "height": 2, "power_rating": 1.0},
            {"width": 4, "height": 2, "power_rating": 2.0},
        ],
    }
    
    return library 