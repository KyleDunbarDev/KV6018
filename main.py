import container_instances
from container_instances import Cylinder, Container, Instance
import random
import math
from typing import List, Dict
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle as PltCircle

class Vector2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Individual:
    """
    Initialise an individual from a list of cylinders

    Args:
        cylinders: A list of Cylinders with id, diameter and weight
    """
    def __init__(self, cylinders: List[Cylinder]):
        self.cylinders = cylinders
        self.ids = [cylinder.id for cylinder in cylinders]
        self.diameters = [cylinder.diameter for cylinder in cylinders]
        self.weights = [cylinder.weight for cylinder in cylinders]
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
            next_x = positions[i].x + prev_r + next_r # Candidate x using bottom-left heuristic. COULD: Candidate x positions beginning from left wall in case of gaps. Not attractive because of weight balancing constraint

            # Check if fits in row
            if next_x + next_r <= container.width:
                # Find min y
                next_y = next_r
                ## Get cylinders in the column of radius
                for j, pos in enumerate(positions):
                    rad_j = radii[j]
                    dx = abs(next_x - pos.x)

                    if dx < (next_r + rad_j): # Only cylinders whose x-range overlaps can block placement
                        dy = math.sqrt((next_r + rad_j) ** 2 - dx ** 2)
                        # Get lowest y that is above all blocking cylinders
                        next_y = max(next_y, pos.y + dy)
                positions.append(Vector2(next_x, next_y))

            else: # Go to next row
                next_x = next_r
                next_y = positions[i].y + prev_r + next_r
                positions.append(Vector2(next_x, next_y))

        # Fitness
        ## Check for overlap
        penalty_overlap = 0
        for i, cyl_i in enumerate(self.cylinders):
            for j, cyl_j  in enumerate(self.cylinders):
                distance = math.sqrt((positions[i].x - positions[j].x)**2 + (positions[i].y - positions[j].y)**2)
                overlap = max(0, (radii[i]+radii[j]) - distance)
                penalty_overlap += overlap**2 # Squared so penalty is proportional to overlap

        ## Check for boundary escape
        penalty_bounds = 0
        for i, cyl in enumerate(self.cylinders):
            upper = max(0, radii[i]+ positions[i].y - container.depth)
            lower = max(0, radii[i] - positions[i].y)
            left = max(0, radii[i] - positions[i].x)
            right = max(0, radii[i] + positions[i].x - container.width)
            penalty_bounds += upper**2 + lower**2 + left**2 + right**2

        ## Check if max weight exceeds capacity. Should always be 0
        penalty_capacity = 0
        penalty_capacity += max(0, sum(self.weights) - container.max_weight)

        ## Check if centre of mass is within 60%
        penalty_CM = 0

        cm_x = 0
        cm_y = 0
        # Calculaye CM along axes
        total_weight = sum(self.weights)
        weighted_x = 0
        weighted_y = 0
        for i, cyl in enumerate(self.cylinders):
            weighted_x += self.weights[i] * positions[i].x
            weighted_y += self.weights[i] * positions[i].y

        if total_weight != 0:
            cm_x = weighted_x / total_weight
            cm_y = weighted_y / total_weight

        # Penalise if CM < 0.2 or CM > 0.8 on each axis
        penalty_CM += max(0, 0.2 * container.width - cm_x)
        penalty_CM += max(0, cm_x - 0.8 * container.width)
        penalty_CM += max(0, 0.2 * container.depth - cm_y)
        penalty_CM += max(0, cm_y - 0.8 * container.depth)
        penalty_CM *= 10


        self.fitness = 0 -(penalty_overlap + penalty_bounds + penalty_capacity + penalty_CM)
        return self.fitness

    def memetic_mutate(self, mutation_rate: float, max_attempts: int, container: Container):
        """
        Local search mutation function.
        Iterates through random swaps and evaluates fitness
            - If fitness is higher, the gene is replaced.
            - Else runs until max_attempts.
        """
        best_fitness = self.fitness

        for _ in range(max_attempts):
            # Choose two positions to swap
            i, j = random.sample(range(len(self.cylinders)), 2)

            # Do swap
            self.cylinders[i], self.cylinders[j] = self.cylinders[j], self.cylinders[i]

            # Update genome attributes
            self.ids = [c.id for c in self.cylinders]
            self.diameters = [c.diameter for c in self.cylinders]
            self.weights = [c.weight for c in self.cylinders]

            # Evaluate new fitness
            new_fitness = self.calculate_fitness(container)

            if new_fitness > best_fitness:
                # Accept improvement
                self.fitness = new_fitness
                best_fitness = new_fitness
            else:
                # Revert swap
                self.cylinders[i], self.cylinders[j] = self.cylinders[j], self.cylinders[i]
                self.ids = [c.id for c in self.cylinders]
                self.diameters = [c.diameter for c in self.cylinders]
                self.weights = [c.weight for c in self.cylinders]

    def mutate(self, mutation_rate: float):
            """
            Random mutation function.
            Swaps genes in the genome regardless of fitness
            """

            if random.random() < mutation_rate:
                i, j = random.sample(range(len(self.cylinders)), 2)
                self.cylinders[i], self.cylinders[j] = self.cylinders[j], self.cylinders[i]

                # Update cached attributes
                self.ids = [c.id for c in self.cylinders]
                self.diameters = [c.diameter for c in self.cylinders]
                self.weights = [c.weight for c in self.cylinders]

    def __str__(self):
        return f"Genes (id): {self.ids}, Fitness: {self.fitness}"


    def draw(self, container, title="Cylinder Placement"):
        """
        Visualise the container and cylinder placements to show fitness.
        """
        # Recompute placement (same as fitness)
        radii = [d / 2 for d in self.diameters]
        positions = []

        positions.append(Vector2(radii[0], radii[0]))

        for i in range(1, len(self.cylinders)):
            prev_r = radii[i - 1]
            next_r = radii[i]

            next_x = positions[i - 1].x + prev_r + next_r

            if next_x + next_r <= container.width:
                next_y = next_r
                for j, pos in enumerate(positions):
                    rad_j = radii[j]
                    dx = abs(next_x - pos.x)
                    if dx < (next_r + rad_j):
                        dy = math.sqrt((next_r + rad_j) ** 2 - dx ** 2)
                        next_y = max(next_y, pos.y + dy)
            else:
                next_x = next_r
                next_y = positions[i - 1].y + prev_r + next_r

            positions.append(Vector2(next_x, next_y))

        fig, ax = plt.subplots(figsize=(6, 6))

        # Container
        container_rect = patches.Rectangle(
            (0, 0),
            container.width,
            container.depth,
            linewidth=2,
            edgecolor="black",
            facecolor="none"
        )
        ax.add_patch(container_rect)

        # Cylinders
        for i, pos in enumerate(positions):
            circle = patches.Circle(
                (pos.x, pos.y),
                radii[i],
                edgecolor="tab:blue",
                facecolor="none",
                linewidth=2
            )
            ax.add_patch(circle)
            ax.text(pos.x, pos.y, str(self.ids[i]),
                    ha="center", va="center", fontsize=9)

        # Centre of Mass
        total_weight = sum(self.weights)
        if total_weight > 0:
            cm_x = sum(self.weights[i] * positions[i].x for i in range(len(self.weights))) / total_weight
            cm_y = sum(self.weights[i] * positions[i].y for i in range(len(self.weights))) / total_weight

            ax.plot(cm_x, cm_y, "rx", markersize=10, label="Centre of Mass")

            # CM safe zone
            ax.add_patch(
                patches.Rectangle(
                    (0.2 * container.width, 0.2 * container.depth),
                    0.6 * container.width,
                    0.6 * container.depth,
                    linestyle="--",
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                    label="CM Safe Zone"
                )
            )

        # Aesthetics
        ax.set_xlim(0, container.width)
        ax.set_ylim(0, container.depth)
        ax.set_aspect("equal")
        ax.set_title(f"Fitness = {self.fitness:.2f}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Depth")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

class Population:
    def __init__(self, pop_size, cylinders: List[Cylinder], container: Container):
        """
        Initialise population of cylinder sequences with given size

        Args:
            pop_size: int of how many individuals to create
            cylinders: list of cylinders with id, diameter, and weight
        """
        self.pop_size = pop_size
        self.cylinders = cylinders
        self.container = container

        gene = self.cylinders.copy() # Copy to shuffle (shuffle shuffles in place)

        self.individuals = []
        for _ in range(pop_size):
            gene = self.cylinders.copy()
            random.shuffle(gene)
            self.individuals.append(Individual(gene))

    def tournament_selection(self, tournament_size = 3) -> Individual:
        """
        Select an individual using tournament selection

        Args:
            tournament_size: Number of selected individuals to showdown in tournament

        Returns:
            Most fit individual
        """

        tournament = random.sample(self.individuals, tournament_size)
        return max(tournament, key=lambda indiv: indiv.fitness)

    def crossover(self, parent1, parent2) -> Individual:
        """
        Order Crossover OX1 to preserve order and ensure no duplication

        Args:
            parent1, parent2: Individual objects
        Returns:
            child_cylinders: List of cylinders in OX1 order
        """
        # Extract genome (list of cylinder IDs)
        genome1 = parent1.ids
        genome2 = parent2.ids
        n = len(genome1)

        # Choose two random cut points
        cut1, cut2 = sorted(random.sample(range(n), 2))

        # Initialise empty child
        child_genome = [None] * n

        # Copy segment from parent1 to child (keeping positions)
        for i in range(cut1, cut2):
            child_genome[i] = genome1[i]

        # Fill remaining positions from parent2, maintaining order
        # Start from cut2 in parent2 and wrap around
        parent2_ptr = cut2 % n
        for i in list(range(cut2, n)) + list(range(0, cut1)):
            if child_genome[i] is None:
                # Find next gene from parent2 not already in child
                while genome2[parent2_ptr] in child_genome:
                    parent2_ptr = (parent2_ptr + 1) % n
                child_genome[i] = genome2[parent2_ptr]
                parent2_ptr = (parent2_ptr + 1) % n

        # Reconstruct cylinders from genome
        cylinders_by_id = {}
        for cylinder in parent1.cylinders:
            cylinders_by_id[cylinder.id] = cylinder

        # Build child cylinders in the crossed over sequence
        child_cylinders = []
        for cylinder_id in child_genome:
            child_cylinders.append(cylinders_by_id[cylinder_id])

        return Individual(child_cylinders)

    def evaluate_population(self):
        """
        Calculates fitness for the current population
        """
        for individual in self.individuals:
            individual.fitness = individual.calculate_fitness(self.container)

    def get_best_individual(self):
        """
        Returns individual with highest fitness
        """
        return max(self.individuals, key=lambda ind: ind.fitness)

    def evolve(self, mutation_rate = 0.01, memetic_attempts = 10, elitism = True):
        """
        Creates next generation through selection, crossover, and mutation.
        Replaces self.individuals in place

        Args:
            mutation_rate:  Chance of mutation per gene
            elitism: if True, most fit individual is guaranteed to be kept
        """

        new_individuals = []

        # Elitism
        if elitism:
            new_individuals.append(self.get_best_individual())

        while len(new_individuals) < self.pop_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            # Breed
            child = self.crossover(parent1, parent2)
            # Mutate
            child.mutate(mutation_rate)
            # Memetic mutate
            child.calculate_fitness(self.container)
            child.memetic_mutate(mutation_rate, memetic_attempts, self.container)

            new_individuals.append(child)

        self.individuals = new_individuals
        self.evaluate_population()

    def get_stats(self):
        """
        Calculates population statistics.
        Should be used after evaluating population.

        Returns:
            Dictionary with 'best', 'average', 'worst' fitness values
        """
        fitnesses = [individual.fitness for individual in self.individuals]
        return {
            'best': max(fitnesses),
            'average': sum(fitnesses) / len(fitnesses),
            'worst': min(fitnesses)
        }

    def print_stats(self):
        stats = self.get_stats()

        out = f"""
            Best: {stats['best']}\n
            Average: {stats['average']}\n
            Worst: {stats['worst']}\n
            """

        print(out)




def main():
    memetic_mutation_attempts = 10
    mutation_rate = 0.01
    population_size = 200
    max_generations = 500

    instance: Instance = container_instances.create_basic_instances()[0]
    population = Population(100, instance.cylinders, instance.container)

    population.evaluate_population()

    for gen in range(max_generations):
        population.evolve(mutation_rate, memetic_mutation_attempts)

        if gen % 20 == 0:
            print(f"------Generation {gen}------")
            population.print_stats()

    best = population.get_best_individual()
    best.draw(instance.container)

if __name__ == "__main__":
    main()
