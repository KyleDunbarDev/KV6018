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
        self.positions = [] # Placement cache

    def calculate_fitness(self, container: Container) -> float:
        """
        Calculate fitness as a numeric value in respect to a given container

        Args:
            container: A Container with width, depth and max_weight
        Returns:
            Fitness value (float)
        """
        # Reset positions
        self.positions = []
        radii = [diameter/2 for diameter in self.diameters]

        # Initialize feasible region (entire container minus boundaries)
        feasible_region = {
            'x_min': 0,
            'x_max': container.width,
            'y_min': 0,
            'y_max': container.depth
        }

        # Try to place each cylinder in sequence
        for i, cyl in enumerate(self.cylinders):
            radius = radii[i]
            placed = False

            # Try multiple candidate positions for feasibility
            candidate_positions = []

            # Generate candidate positions based on loading from rear (y=0)
            # Try positions along the rear wall first, then expand
            grid_step = radius / 2  # Grid for candidate positions

            # If first cylinder, try to place near rear wall
            if i == 0:
                # Start from rear (y = radius to avoid boundary violation)
                for y_offset in [radius, radius*2, radius*3]:
                    for x in np.arange(radius, container.width - radius, grid_step):
                        candidate_positions.append(Vector2(x, y_offset))

            # For subsequent cylinders, try positions that:
            # 1. Are near already placed cylinders (to pack efficiently)
            # 2. Are along the current "frontier" of placed cylinders
            else:
                # Generate grid of candidate positions
                for y in np.arange(radius, container.depth - radius, grid_step):
                    for x in np.arange(radius, container.width - radius, grid_step):
                        candidate_positions.append(Vector2(x, y))

                # Also try positions tangent to already placed cylinders
                for j, existing_pos in enumerate(self.positions[:i]):
                    existing_radius = radii[j]
                    # Try positions around existing cylinder
                    for angle in np.linspace(0, 2*math.pi, 16):
                        dx = math.cos(angle) * (radius + existing_radius)
                        dy = math.sin(angle) * (radius + existing_radius)
                        candidate = Vector2(existing_pos.x + dx, existing_pos.y + dy)
                        candidate_positions.append(candidate)

            random.seed(42)
            # Shuffle candidates to avoid bias
            random.shuffle(candidate_positions)

            # Try candidates in order
            for pos in candidate_positions:
                if self.is_position_feasible(pos, radius, i, radii, container):
                    self.positions.append(pos)
                    placed = True
                    break

            # If no feasible position found, use a fallback position with penalty
            if not placed:
                # Place at a default position (will incur heavy penalty)
                fallback_pos = Vector2(
                    container.width/2 + random.uniform(-1, 1),
                    container.depth/2 + random.uniform(-1, 1)
                )
                self.positions.append(fallback_pos)

        # Calculate fitness penalties
        return self._calculate_penalties(radii, container)

    def is_position_feasible(self, pos: Vector2, radius: float, current_idx: int, radii: List[float], container: Container) -> bool:
        """
        Check if a position is feasible for the current cylinder.
        """
        # Check container boundaries
        if (pos.x - radius < 0 or pos.x + radius > container.width or
            pos.y - radius < 0 or pos.y + radius > container.depth):
            return False

        # Check overlap with already placed cylinders
        for j in range(current_idx):
            existing_pos = self.positions[j]
            existing_radius = radii[j]
            distance = math.sqrt((pos.x - existing_pos.x)**2 +
                                (pos.y - existing_pos.y)**2)
            if distance < (radius + existing_radius - 1e-6):  # Small tolerance
                return False

        return True

    def _calculate_penalties(self, radii: List[float], container: Container) -> float:
        """
        Calculate all penalty components.
        """
        # Penalty for overlap
        penalty_overlap = 0
        n = len(self.cylinders)
        for i in range(n):
            for j in range(i + 1, n):
                distance = math.sqrt(
                    (self.positions[i].x - self.positions[j].x)**2 +
                    (self.positions[i].y - self.positions[j].y)**2
                )
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    overlap = min_distance - distance
                    penalty_overlap += overlap**2

        # Penalty for boundary violation
        penalty_bounds = 0
        for i, pos in enumerate(self.positions):
            radius = radii[i]
            # Check each boundary
            if pos.x - radius < 0:
                penalty_bounds += (radius - pos.x)**2
            if pos.x + radius > container.width:
                penalty_bounds += (pos.x + radius - container.width)**2
            if pos.y - radius < 0:
                penalty_bounds += (radius - pos.y)**2
            if pos.y + radius > container.depth:
                penalty_bounds += (pos.y + radius - container.depth)**2

        # Penalty for weight capacity
        total_weight = sum(self.weights)
        penalty_capacity = max(0, total_weight - container.max_weight)**2

        # Penalty for center of mass outside central 60%
        penalty_CM = 0
        reward_CM = 0

        if total_weight > 0:
            cm_x = sum(self.weights[i] * self.positions[i].x for i in range(n)) / total_weight
            cm_y = sum(self.weights[i] * self.positions[i].y for i in range(n)) / total_weight

            # Distance from center (normalized)
            centre_x = container.width/2
            centre_y = container.depth/2
            distance_from_center_x = abs(cm_x - centre_x) / (container.width / 2)
            distance_from_center_y = abs(cm_y - centre_y) / (container.depth / 2)
            # Reward for being close to centre
            reward_CM = 100 * (1 - distance_from_center_x) * (1 - distance_from_center_y)

            # Center should be within 20% to 80% of container dimensions
            if cm_x < 0.2 * container.width:
                penalty_CM += (0.2 * container.width - cm_x)**2
            if cm_x > 0.8 * container.width:
                penalty_CM += (cm_x - 0.8 * container.width)**2
            if cm_y < 0.2 * container.depth:
                penalty_CM += (0.2 * container.depth - cm_y)**2
            if cm_y > 0.8 * container.depth:
                penalty_CM += (cm_y - 0.8 * container.depth)**2
            # Reward for CM being close to centre
            centre = Vector2(container.width/2, container.depth/2)
            dist = math.sqrt((centre.x - cm_x)**2
                + (centre.y - cm_y)**2 )
            penalty_CM -= dist
        else:
            penalty_CM = 0
            reward_CM = 0


        # Total penalty (negative for fitness maximization)
        total_penalty = (
            penalty_overlap * 10.0 +
            penalty_bounds * 5.0 +
            penalty_capacity * 100.0 +
            penalty_CM * 2.0
        )

        self.fitness = -total_penalty + reward_CM
        return self.fitness

    def _calculate_detailed_penalties(self, container: Container) -> dict:
        """
        Calculate and return detailed penalty breakdown.
        """
        if not self.positions:
            self.calculate_fitness(container)

        radii = [d/2 for d in self.diameters]
        n = len(self.cylinders)

        # Initialize penalties
        penalty_overlap = 0
        penalty_bounds = 0

        # Boundary penalties
        for i, pos in enumerate(self.positions):
            radius = radii[i]
            if pos.x - radius < 0:
                penalty_bounds += (radius - pos.x)**2
            if pos.x + radius > container.width:
                penalty_bounds += (pos.x + radius - container.width)**2
            if pos.y - radius < 0:
                penalty_bounds += (radius - pos.y)**2
            if pos.y + radius > container.depth:
                penalty_bounds += (pos.y + radius - container.depth)**2

        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                distance = math.sqrt(
                    (self.positions[i].x - self.positions[j].x)**2 +
                    (self.positions[i].y - self.positions[j].y)**2
                )
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    overlap = min_distance - distance
                    penalty_overlap += overlap**2

        # Capacity penalty
        total_weight = sum(self.weights)
        penalty_capacity = max(0, total_weight - container.max_weight)**2

        # Center of mass calculations
        center_mass_penalty = 0
        center_mass_reward = 0
        center_mass = None

        if total_weight > 0:
            cm_x = sum(self.weights[i] * self.positions[i].x for i in range(n)) / total_weight
            cm_y = sum(self.weights[i] * self.positions[i].y for i in range(n)) / total_weight
            center_mass = (cm_x, cm_y)

            # Penalty for being outside safe zone (20%-80%)
            if cm_x < 0.2 * container.width:
                center_mass_penalty += (0.2 * container.width - cm_x)**2
            if cm_x > 0.8 * container.width:
                center_mass_penalty += (cm_x - 0.8 * container.width)**2
            if cm_y < 0.2 * container.depth:
                center_mass_penalty += (0.2 * container.depth - cm_y)**2
            if cm_y > 0.8 * container.depth:
                center_mass_penalty += (cm_y - 0.8 * container.depth)**2

            # Reward for being close to center
            target_x = container.width / 2
            target_y = container.depth / 2
            distance_from_center = math.sqrt((cm_x - target_x)**2 + (cm_y - target_y)**2)
            max_distance = math.sqrt((container.width/2)**2 + (container.depth/2)**2)
            center_mass_reward = 100 * (1 - distance_from_center / max_distance)

        # Total weighted penalties
        total_penalty = (
            penalty_overlap * 10.0 +
            penalty_bounds * 5.0 +
            penalty_capacity * 100.0 +
            center_mass_penalty * 2.0
        )

        return {
            'overlap': penalty_overlap * 10.0,
            'boundary': penalty_bounds * 5.0,
            'capacity': penalty_capacity * 100.0,
            'center_mass_penalty': center_mass_penalty * 2.0,
            'center_mass_reward': center_mass_reward,
            'total_penalty': total_penalty,
            'center_mass': center_mass,
            'total_weight': total_weight
        }

    def mutate(self, mutation_rate: float):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(self.cylinders)), 2)
            self.cylinders[i], self.cylinders[j] = self.cylinders[j], self.cylinders[i]

            # Update cached attributes
            self.ids = [c.id for c in self.cylinders]
            self.diameters = [c.diameter for c in self.cylinders]
            self.weights = [c.weight for c in self.cylinders]
            self.positions = []  # Reset positions since genome changed

    def memetic_mutate(self, mutation_rate: float, max_attempts: int, container: Container):
        best_fitness = self.fitness
        best_positions = self.positions.copy() if self.positions else []

        for _ in range(max_attempts):
            # Choose two positions to swap
            i, j = random.sample(range(len(self.cylinders)), 2)

            # Do swap
            self.cylinders[i], self.cylinders[j] = self.cylinders[j], self.cylinders[i]

            # Update genome attributes
            self.ids = [c.id for c in self.cylinders]
            self.diameters = [c.diameter for c in self.cylinders]
            self.weights = [c.weight for c in self.cylinders]
            self.positions = []  # Reset positions

            # Evaluate new fitness
            new_fitness = self.calculate_fitness(container)

            if new_fitness > best_fitness:
                # Accept improvement
                self.fitness = new_fitness
                best_fitness = new_fitness
                best_positions = self.positions.copy()
            else:
                # Revert swap
                self.cylinders[i], self.cylinders[j] = self.cylinders[j], self.cylinders[i]
                self.ids = [c.id for c in self.cylinders]
                self.diameters = [c.diameter for c in self.cylinders]
                self.weights = [c.weight for c in self.cylinders]

        # Restore best positions
        self.positions = best_positions

    def __str__(self):
        return f"Genes (id): {self.ids}, Fitness: {self.fitness}"


    def draw(self, container, title="Cylinder Placement"):
        """
        Visualise the container and cylinder placements to show fitness.
        """

        if not self.positions:
            self.calculate_fitness(container)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw container
        container_rect = patches.Rectangle(
            (0, 0),
            container.width,
            container.depth,
            linewidth=2,
            edgecolor="black",
            facecolor="none"
        )
        ax.add_patch(container_rect)

        # Draw cylinders
        radii = [d/2 for d in self.diameters]
        for i, (pos, radius) in enumerate(zip(self.positions, radii)):
            circle = patches.Circle(
                (pos.x, pos.y),
                radius,
                edgecolor="tab:blue",
                facecolor="lightblue",
                alpha=0.7,
                linewidth=2
            )
            ax.add_patch(circle)
            # Add cylinder ID and weight
            ax.text(pos.x, pos.y, f"{self.ids[i]}\n{self.weights[i]}kg",
                   ha="center", va="center", fontsize=8, fontweight='bold')

        # Draw center of mass and safe zone
        total_weight = sum(self.weights)
        if total_weight > 0:
            cm_x = sum(self.weights[i] * self.positions[i].x for i in range(len(self.weights))) / total_weight
            cm_y = sum(self.weights[i] * self.positions[i].y for i in range(len(self.weights))) / total_weight

            # Center of mass marker
            ax.plot(cm_x, cm_y, "rx", markersize=12, markeredgewidth=2, label="Center of Mass")

            # Safe zone (central 60%)
            safe_zone = patches.Rectangle(
                (0.2 * container.width, 0.2 * container.depth),
                0.6 * container.width,
                0.6 * container.depth,
                linestyle="--",
                linewidth=1,
                edgecolor="green",
                facecolor="none",
                alpha=0.5,
                label="CM Safe Zone"
            )
            ax.add_patch(safe_zone)

        # Set up the plot
        ax.set_xlim(0, container.width)
        ax.set_ylim(0, container.depth)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nFitness: {self.fitness:.2f}")
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.3)
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
            # child.calculate_fitness(self.container)
            # child.memetic_mutate(mutation_rate, memetic_attempts, self.container)

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


def run_single_instance(instance, mutation_rate=0.01, memetic_attempts=10, population_size=200, max_generations=500, print_interval=20, draw_result=True):
    """
    Run the GA on a single instance with visualisation.

    Args:
        instance: Instance to solve with name, container, and list of cylinders
        mutation_rate: Probability of mutation per gene
        memetic_attempts: Number of local search attempts before stopping
        population_size: Size of population
        max_generations: Maximum number of generations to evolve
        print_interval: Print stats every N generations (0 to disable)
        draw_result: If True, visualise the best solution

    Returns:
        Dictionary containing the best individual and statistics
    """
    # Initialise population
    population = Population(population_size, instance.cylinders, instance.container)
    population.evaluate_population()

    # Evolution
    for gen in range(max_generations):
        population.evolve(mutation_rate, memetic_attempts)

        if print_interval > 0 and gen % print_interval == 0:
            print(f"------Generation {gen}------")
            population.print_stats()

    # Get best solution
    best = population.get_best_individual()
    # Visualise
    if draw_result:
        best.draw(instance.container)

    return {
        'best_individual': best,
        'final_stats': population.get_stats(),
        'instance': instance
    }

def run_all_instances(mutation_rate=0.01, memetic_attempts=10, population_size=200, max_generations=500, print_interval=50, verbose=True):
    """
    Run the GA on all available instances and analyse results.

    Args:
        mutation_rate: Probability of mutation per gene
        memetic_attempts: Number of local search attempts before stopping
        population_size: Size of population
        max_generations: Maximum number of generations to evolve
        verbose: If True, prints progress during evolution

    Returns:
        Dictionary containing results for each instance
    """
    # Gather all instances
    all_instances = []
    basic = container_instances.create_basic_instances()
    challenging = container_instances.create_challenging_instances()

    all_instances.extend([("Basic", inst) for inst in basic])
    all_instances.extend([("Challenging", inst) for inst in challenging])

    results = []

    print("=" * 80)
    print("RUNNING GENETIC ALGORITHM ON ALL INSTANCES")
    print("=" * 80)
    print(f"Parameters: pop_size={population_size}, generations={max_generations}")
    print(f"            mutation_rate={mutation_rate}, memetic_attempts={memetic_attempts}")
    print("=" * 80)
    print()

    for category, instance in all_instances:
        print(f"\n{'=' * 80}")
        print(f"Instance: {instance.name} ({category})")
        print(f"Container: {instance.container.width}m × {instance.container.depth}m, "
              f"max weight: {instance.container.max_weight}kg")
        print(f"Cylinders: {len(instance.cylinders)}")
        print(f"{'=' * 80}")

        # Initialise population
        population = Population(population_size, instance.cylinders, instance.container)
        population.evaluate_population()

        initial_stats = population.get_stats()

        # Evolution
        for gen in range(max_generations):
            population.evolve(mutation_rate, memetic_attempts)

            if verbose and gen % print_interval == 0:
                stats = population.get_stats()
                print(f"Generation {gen:4d}: Best={stats['best']:8.2f}, "
                      f"Avg={stats['average']:8.2f}, Worst={stats['worst']:8.2f}")

        # Final statistics
        final_stats = population.get_stats()
        best_individual = population.get_best_individual()

        # Determine success (fitness >= -0.01 is considered successful - small tolerance for numerical errors)
        is_successful = final_stats['best'] >= -0.01

        print(f"\n{'=' * 40}")
        print(f"FINAL RESULTS:")
        print(f"  Initial Best Fitness: {initial_stats['best']:.4f}")
        print(f"  Final Best Fitness:   {final_stats['best']:.4f}")
        print(f"  Final Avg Fitness:    {final_stats['average']:.4f}")
        print(f"  Improvement:          {final_stats['best'] - initial_stats['best']:.4f}")
        print(f"  Status:               {'SUCCESS' if is_successful else 'FAILED'}")
        print(f"{'=' * 40}")

        # Store results
        results.append({
            'category': category,
            'name': instance.name,
            'instance': instance,
            'best_individual': best_individual,
            'initial_fitness': initial_stats['best'],
            'final_fitness': final_stats['best'],
            'avg_fitness': final_stats['average'],
            'improvement': final_stats['best'] - initial_stats['best'],
            'is_successful': is_successful,
            'num_cylinders': len(instance.cylinders),
            'container_area': instance.container.width * instance.container.depth
        })

        # Draw the best solution
        best_individual.draw(instance.container,
                            title=f"{instance.name} - Fitness: {final_stats['best']:.2f}")

    # Print summary report
    print("\n\n" + "=" * 80)
    print("SUMMARY REPORT - ALL INSTANCES")
    print("=" * 80)

    # Sort by success (successful first) then by fitness (descending)
    sorted_results = sorted(results,
                           key=lambda x: (x['is_successful'], x['final_fitness']),
                           reverse=True)

    successful = [r for r in sorted_results if r['is_successful']]
    failed = [r for r in sorted_results if not r['is_successful']]

    print(f"\nSuccessful Instances: {len(successful)}/{len(results)}")
    print(f"Failed Instances:     {len(failed)}/{len(results)}")

    if successful:
        print("\n" + "-" * 80)
        print("SUCCESSFUL INSTANCES (sorted by fitness, descending):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Instance Name':<35} {'Fitness':<12} {'Cylinders':<12}")
        print("-" * 80)
        for i, result in enumerate(successful, 1):
            print(f"{i:<6} {result['name']:<35} {result['final_fitness']:>10.4f}  "
                  f"{result['num_cylinders']:>10}")

    if failed:
        print("\n" + "-" * 80)
        print("FAILED INSTANCES (sorted by fitness, descending):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Instance Name':<35} {'Fitness':<12} {'Cylinders':<12}")
        print("-" * 80)
        for i, result in enumerate(failed, 1):
            print(f"{i:<6} {result['name']:<35} {result['final_fitness']:>10.4f}  "
                  f"{result['num_cylinders']:>10}")

    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE METRICS")
    print("=" * 80)
    print(f"{'Instance Name':<35} {'Initial':<12} {'Final':<12} {'Improvement':<12} {'Status':<10}")
    print("-" * 80)
    for result in sorted_results:
        status = "SUCCESS" if result['is_successful'] else "FAILED"
        print(f"{result['name']:<35} {result['initial_fitness']:>10.2f}  "
              f"{result['final_fitness']:>10.2f}  {result['improvement']:>10.2f}  {status:<10}")

    print("=" * 80)

    return {
        'all_results': sorted_results,
        'successful': successful,
        'failed': failed,
        'success_rate': len(successful) / len(results) if results else 0
    }

def evaluate_and_visualize_sequence(instance, id_sequence, show_visualization=True):
    """
    Evaluate and visualize a specific sequence of cylinder IDs.

    Args:
        instance: The problem instance (from container_instances)
        id_sequence: List of cylinder IDs in the order to place them
        show_visualization: If True, show the visualization plot

    Returns:
        Dictionary with fitness, positions, and all penalty components
    """
    # Create a dictionary to map IDs to cylinders
    cylinder_by_id = {cyl.id: cyl for cyl in instance.cylinders}

    # Create cylinders in the specified sequence order
    ordered_cylinders = []
    for cyl_id in id_sequence:
        if cyl_id in cylinder_by_id:
            ordered_cylinders.append(cylinder_by_id[cyl_id])
        else:
            raise ValueError(f"Cylinder ID {cyl_id} not found in instance")

    # Create an Individual with this sequence
    individual = Individual(ordered_cylinders)

    # Calculate fitness
    fitness = individual.calculate_fitness(instance.container)

    # Calculate detailed penalty breakdown
    penalty_breakdown = individual._calculate_detailed_penalties(instance.container)

    # Print results
    print("=" * 70)
    print(f"SEQUENCE EVALUATION - {instance.name}")
    print("=" * 70)
    print(f"Sequence: {id_sequence}")
    print(f"Total cylinders: {len(id_sequence)}")
    print(f"Fitness: {fitness:.4f}")
    print("\nPenalty Breakdown:")
    print(f"  Overlap penalty:      {penalty_breakdown['overlap']:.4f}")
    print(f"  Boundary penalty:     {penalty_breakdown['boundary']:.4f}")
    print(f"  Capacity penalty:     {penalty_breakdown['capacity']:.4f}")
    print(f"  Center of Mass penalty: {penalty_breakdown['center_mass_penalty']:.4f}")
    print(f"  Center of Mass reward:  {penalty_breakdown['center_mass_reward']:.4f}")
    print(f"  Total penalty:        {penalty_breakdown['total_penalty']:.4f}")

    # Print placement information
    print("\nPlacement Details:")
    print(f"{'ID':<4} {'Diameter':<8} {'Weight':<8} {'X':<8} {'Y':<8}")
    print("-" * 40)
    for i, cyl in enumerate(individual.cylinders):
        if i < len(individual.positions):
            pos = individual.positions[i]
            print(f"{cyl.id:<4} {cyl.diameter:<8.2f} {cyl.weight:<8.2f} {pos.x:<8.2f} {pos.y:<8.2f}")
        else:
            print(f"{cyl.id:<4} {cyl.diameter:<8.2f} {cyl.weight:<8.2f} {'N/A':<8} {'N/A':<8}")

    # Show center of mass information
    if penalty_breakdown['center_mass'] is not None:
        cm_x, cm_y = penalty_breakdown['center_mass']
        print(f"\nCenter of Mass: ({cm_x:.2f}, {cm_y:.2f})")
        print(f"Container Center: ({instance.container.width/2:.2f}, {instance.container.depth/2:.2f})")
        print(f"Distance from center: {math.sqrt((cm_x - instance.container.width/2)**2 + (cm_y - instance.container.depth/2)**2):.2f}")

    # Total weight check
    total_weight = sum(cyl.weight for cyl in individual.cylinders)
    print(f"\nTotal weight: {total_weight:.2f} / {instance.container.max_weight:.2f}")
    print(f"Weight utilization: {(total_weight/instance.container.max_weight*100):.1f}%")

    # Show visualization
    if show_visualization:
        individual.draw(instance.container,
                        title=f"Sequence: {id_sequence[:5]}{'...' if len(id_sequence) > 5 else ''}")

    return {
        'fitness': fitness,
        'positions': individual.positions.copy() if individual.positions else [],
        'penalty_breakdown': penalty_breakdown,
        'individual': individual
    }

def main():
    memetic_mutation_attempts = 10
    mutation_rate = 0.01
    population_size = 200
    max_generations = 200


    # You can choose to run a single instance with visualation or all instances with a report & visualisations
    # by commenting and uncommenting the respective options



    #! Option 1: Run single given instance
    ## Choose instance
    # instance: Instance = container_instances.create_basic_instances()[0]
    # # instance: Instance = container_instances.create_challenging_instances()[0]
    #
    # result = run_single_instance(instance=instance, mutation_rate=mutation_rate, memetic_attempts=memetic_mutation_attempts, population_size=population_size, max_generations=max_generations, print_interval=20, draw_result=True)
    # print(f"\nFinal fitness: {result['final_stats']['best']:.4f}")
    # print(f"Best sequence (cylinder IDs): {result['best_individual'].ids}")
    # print(f"Best sequence (full details):")
    # for i, cyl in enumerate(result['best_individual'].cylinders):
    #     print(f"  Position {i+1}: ID={cyl.id}, Diameter={cyl.diameter}, Weight={cyl.weight}")



    # #! Option 2: Run all instances
    # results = run_all_instances( mutation_rate=mutation_rate, memetic_attempts=memetic_mutation_attempts, population_size=population_size, max_generations=max_generations, verbose=True)
    # print(f"\nOverall Success Rate: {results['success_rate']*100:.1f}%")
    # print("\n" + "=" * 80)
    # print("BEST SEQUENCES FOR ALL INSTANCES")
    # print("=" * 80)
    # for result in results['all_results']:
    #     print(f"\n{result['name']} (Fitness: {result['final_fitness']:.4f}):")
    #     print(f"  Sequence: {result['best_individual'].ids}")
    #     status = "SUCCESS" if result['is_successful'] else "FAILED"
    #     print(f"  Status: {status}")



    #! Option 3: Test specific sequences
    print("\nOption 3: Testing specific sequences")
    basic_instances = container_instances.create_basic_instances()
    instance = basic_instances[2]  # Use the first basic instance
    # Test a specific sequence
    sequence = [1, 2, 3, 4, 5]  # NOTE: Ensure ids used are present in instance
    result = evaluate_and_visualize_sequence(instance, sequence, show_visualization=True)

if __name__ == "__main__":
    main()
