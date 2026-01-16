from container_instances import Cylinder, Container
import random
import math

class Vector2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Individual:
    """
    Initialise an individual from a list of cylinders which gets shuffled

    Args:
        cylinders: A list of Cylinders with diameter and weight
    """
    def __init__(self, cylinders):
        self.cylinders = cylinders
        self.num_genes = len(cylinders)
        self.ids = [cylinder['id'] for cylinder in cylinders]
        self.diameters = [cylinder['diameter'] for cylinder in cylinders]
        self.weights = [cylinder['weight'] for cylinder in cylinders]

        shuffled_ids = self.ids.copy()
        random.shuffle(shuffled_ids)
        self.genes = shuffled_ids

        self.fitness = 0

def calculate_fitness(self, container: Container) -> float:
    """
    Calculate fitness as a numeric value in respect to a given container

    Args:
        container: A Container with width, depth and max_weight
    Returns:
        Fitness value (float)
    """
    # Placement
    radii = [diameter/2 for diameter in self.diameters]
    positions: list[Vector2] = []
    ## Place first cylinder
    positions.append(Vector2(0+radii[0], 0+radii[0]))

    ## Place subsequent cylinders
    for i, cyl in enumerate(self.cylinders[1:]):
        # Get previous and next cylinders' radii. Find next x
        prev_r = radii[i]
        next_r = radii[i+1]
        next_x = positions[i].x + prev_r + next_r # Candidate x using bottom-left heuristic

        # Check if fits in row
        if next_x + next_r <= container.width:
            # Calculate y and check validate for protrusion of previous cylinders
            next_y = next_r
            ## Get cylinders in the column of radius
            for j, pos in enumerate(positions):
                rad_j = radii[j]
                dx = abs(next_x - pos.x)

                # Only cylinders whose x-range overlaps can block placement
                if dx < (next_r + rad_j):
                    dy = math.sqrt((next_r + rad_j) ** 2 - dx ** 2)
                    next_y = max(next_y, pos.y + dy)
            positions.append(Vector2(next_x, next_y))

        else: # Go to next row
            next_x = next_r
            next_y = positions[i].y + prev_r + next_r
            positions.append(Vector2(next_x, next_y))


    # Fitness
    ## Check for overlap

    ## Check for boundary escape

    ## Check if max weight exceeds capacity

    ## Check if centre of mass is within 60%



    return 0.0

def mutate(self, mutation_rate: float, max_attempts: int):
    """
    Local search mutation function.
    Iterates through random swaps and evaluates fitness
        - If fitness is higher, the gene is replaced.
        - Else runs until max_attempts.
    """

def __str__(self):
    return f"Genes: {self.genes}, Fitness: {self.fitness}"


def main():
    pass

if __name__ == "__main__":
    main()
