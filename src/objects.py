import numpy as np
from typing import Set, Tuple, List

class Object:
    """Base class for objects that can be placed in the grid world."""
    
    def __init__(self, cells: Set[Tuple[int, int]]):
        """
        Initialize an object with a set of cells it occupies.
        
        Args:
            cells: A set of (x, y) coordinates representing the cells occupied by the object.
        """
        # Normalize the object position to have minimum x and y at (0, 0)
        min_x = min(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        self.cells = {(x - min_x, y - min_y) for x, y in cells}
    
    def get_cells(self) -> Set[Tuple[int, int]]:
        """Return the set of cells occupied by the object."""
        return self.cells
    
    def get_cells_at_position(self, pos_x: int, pos_y: int) -> Set[Tuple[int, int]]:
        """
        Return the set of cells occupied by the object when placed at the given position.
        
        Args:
            pos_x: The x-coordinate of the position to place the object.
            pos_y: The y-coordinate of the position to place the object.
            
        Returns:
            A set of (x, y) coordinates representing the cells occupied by the object.
        """
        return {(x + pos_x, y + pos_y) for x, y in self.cells}
    
    def rotate_90(self) -> 'Object':
        """
        Return a new object rotated 90 degrees clockwise.
        
        Returns:
            A new Object instance representing the rotated object.
        """
        # For a 90-degree rotation, (x, y) -> (-y, x)
        rotated_cells = {(-y, x) for x, y in self.cells}
        return Object(rotated_cells)
    
    def rotate_180(self) -> 'Object':
        """
        Return a new object rotated 180 degrees.
        
        Returns:
            A new Object instance representing the rotated object.
        """
        # For a 180-degree rotation, (x, y) -> (-x, -y)
        rotated_cells = {(-x, -y) for x, y in self.cells}
        return Object(rotated_cells)
    
    def rotate_270(self) -> 'Object':
        """
        Return a new object rotated 270 degrees clockwise (or 90 degrees counterclockwise).
        
        Returns:
            A new Object instance representing the rotated object.
        """
        # For a 270-degree rotation, (x, y) -> (y, -x)
        rotated_cells = {(y, -x) for x, y in self.cells}
        return Object(rotated_cells)
    
    def get_all_rotations(self) -> List['Object']:
        """
        Return a list of all possible rotations of this object.
        
        Returns:
            A list of Object instances representing all possible rotations.
        """
        rotations = [self]
        for _ in range(3):  # Add 90, 180, and 270 degree rotations
            rotations.append(rotations[-1].rotate_90())
        
        # Remove duplicates (some rotations may be the same for symmetrical objects)
        unique_rotations = []
        seen_cells = set()
        
        for obj in rotations:
            cells_tuple = tuple(sorted(obj.cells))
            if cells_tuple not in seen_cells:
                seen_cells.add(cells_tuple)
                unique_rotations.append(obj)
        
        return unique_rotations


class Square(Object):
    """A square object with specified size."""
    
    def __init__(self, size: int = 2):
        """
        Initialize a square with the given size.
        
        Args:
            size: The side length of the square.
        """
        cells = {(x, y) for x in range(size) for y in range(size)}
        super().__init__(cells)


class Rectangle(Object):
    """A rectangular object with specified width and height."""
    
    def __init__(self, width: int = 2, height: int = 3):
        """
        Initialize a rectangle with the given width and height.
        
        Args:
            width: The width of the rectangle.
            height: The height of the rectangle.
        """
        cells = {(x, y) for x in range(width) for y in range(height)}
        super().__init__(cells)


class LShape(Object):
    """An L-shaped object."""
    
    def __init__(self, size: int = 3):
        """
        Initialize an L-shaped object with the given size.
        
        Args:
            size: The length of each arm of the L.
        """
        cells = {(0, y) for y in range(size)} | {(x, 0) for x in range(1, size)}
        super().__init__(cells)


class TShape(Object):
    """A T-shaped object."""
    
    def __init__(self, width: int = 3, height: int = 2):
        """
        Initialize a T-shaped object with the given dimensions.
        
        Args:
            width: The width of the top of the T.
            height: The height of the stem of the T.
        """
        cells = {(x, 0) for x in range(width)} | {(width // 2, y) for y in range(1, height + 1)}
        super().__init__(cells) 