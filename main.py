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

random.seed(42)

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


    def place_cylinders(self, radii: List[float], container: Container):
        """
        Places cylinders memetically while enforcing loading order.

        Args:
            radii: A list of cylinder's radii
        """

        fpt = 1e-6 # Floating point tolerance
        max_candidates = 80
        x_samples = 20
        y_samples = 4
        max_slack_ratio = 0.4   # Fraction of container depth allowed as vertical slack

        self.positions = []
        n = len(radii)

        # Centre of mass tracking
        total_weight = 0.0
        com_y = 0.0
        cx = container.width / 2
        target_y = container.depth / 2

        for i in range(n):
            radius = radii[i]
            weight = self.cylinders[i].weight if hasattr(self, "cylinders") else 1.0
            candidates = []

            fill_ratio = i / n
            max_slack = max_slack_ratio * (1.0 - fill_ratio) * container.depth # Slack shrinks as container fills

            # 1. Frontier-based x sampling (loading-order aware)
            for x in np.linspace(radius, container.width - radius, x_samples):
                # Loading-order frontier
                min_y = radius
                for j in range(i):
                    prev = self.positions[j]
                    prev_r = radii[j]
                    if abs(x - prev.x) < radius + prev_r:
                        min_y = max(min_y, prev.y)

                if min_y + radius > container.depth:
                    continue

                y_low = min_y
                y_high = min(container.depth - radius, min_y + max_slack)

                for y in np.linspace(y_low, y_high, y_samples):
                    candidates.append(Vector2(x, y))
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

            # 2. Tangent candidates around existing cylinders
            for j in range(i):
                prev = self.positions[j]
                prev_r = radii[j]

                for angle in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                    cx_t = prev.x + math.cos(angle) * (radius + prev_r)
                    cy_t = prev.y + math.sin(angle) * (radius + prev_r)

                    if (cx_t - radius < 0 or cx_t + radius > container.width or
                        cy_t - radius < 0 or cy_t + radius > container.depth):
                        continue

                    candidates.append(Vector2(cx_t, cy_t))
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

            # 3. Feasibilty + balance aware selection
            best_pos = None
            best_score = float("inf")
            best_new_com_y = 0.0
            best_new_total_weight = 0.0

            for pos in candidates:
                feasible = True

                for j in range(i):
                    prev = self.positions[j]
                    prev_r = radii[j]

                    dx = pos.x - prev.x
                    dy = pos.y - prev.y

                    # Overlap constraint
                    if dx * dx + dy * dy < (radius + prev_r) ** 2 - fpt:
                        feasible = False
                        break

                    # Loading order constraint
                    if abs(dx) < radius + prev_r and dy < -fpt:
                        feasible = False
                        break

                if not feasible:
                    continue

                # Predict centre of mass after placement
                new_total_weight = total_weight + weight
                new_com_y = (
                    (com_y * total_weight + weight * pos.y) / new_total_weight
                    if total_weight > 0
                    else pos.y
                )

                # Balance-aware scoring
                score = (
                    2.0 * abs(pos.x - cx) +                 # horizontal balance
                    3.0 * abs(new_com_y - target_y) +       # CoM correction
                    0.5 * pos.y                              # mild compactness
                )

                if score < best_score:
                    best_score = score
                    best_pos = pos
                    best_new_com_y = new_com_y
                    best_new_total_weight = new_total_weight

            # 4. Place best candidate or fallback
            if best_pos is not None:
                self.positions.append(best_pos)
                total_weight = best_new_total_weight
                com_y = best_new_com_y
            else:
                # Fallback placement (penalised naturally)
                fallback = Vector2(
                    container.width / 2 + random.uniform(-1, 1),
                    container.depth / 2 + random.uniform(-1, 1)
                )
                self.positions.append(fallback)
                total_weight += weight
                com_y = (
                    (com_y * (total_weight - weight) + weight * fallback.y) / total_weight
                )

        return self.positions


    def calculate_fitness(self, container: Container) -> float:
        """
        Gets the calculated fitness as a numeric value in respect to a given container.
        Fitness is not comparable between differently n number cylinders as rewards are given per cylinder.

        Args:
            container: A Container with width, depth and max_weight
        Returns:
            Fitness value (float)
        """
        return self.evaluate(container)["fitness"]

    def evaluate(self, container: Container) -> Dict:
        """
        Calculates all penalty and reward components.
        First places cylinders if not placed, and then evaluates all penalties and rewards.
        Returns a dict of evaluated metrics

        Args:
            container: The container to evaluate self against with width, depth, and max_weight
        Returns:
            Dict of metrics:
                "fitness": float,
                "penalties": Dict,
                "rewards": Dict,
                "total_weight": float,
                "centre_of_mass": Vector2,
                "is_successful": bool,
        """

        if not self.positions:
            radii = [d / 2 for d in self.diameters]
            self.place_cylinders(radii, container)
        else:
            radii = [d / 2 for d in self.diameters]

        n = len(self.cylinders)

        penalties = {
            "overlap": 0.0,
            "bounds": 0.0,
            "capacity": 0.0,
            "centre_of_mass": 0.0,
            "loading_order": 0.0,
        }

        rewards = {
            "centre_mass_reward": 0.0,
            "cylinder_centrality": 0.0,
        }

        # Overlap + loading order
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.positions[i].x - self.positions[j].x
                dy = self.positions[i].y - self.positions[j].y
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    penalties["overlap"] += (min_dist - dist) ** 2

                # Loading order: later cylinder cannot block earlier one
                if self.positions[i].y < self.positions[j].y:
                    if abs(dx) < min_dist:
                        penalties["loading_order"] += 1.0

        # Boundary
        for i, pos in enumerate(self.positions):
            r = radii[i]
            if pos.x - r < 0:
                penalties["bounds"] += (r - pos.x) ** 2
            if pos.x + r > container.width:
                penalties["bounds"] += (pos.x + r - container.width) ** 2
            if pos.y - r < 0:
                penalties["bounds"] += (r - pos.y) ** 2
            if pos.y + r > container.depth:
                penalties["bounds"] += (pos.y + r - container.depth) ** 2

        # Weight capacity
        total_weight = sum(self.weights)
        if total_weight > container.max_weight:
            penalties["capacity"] = (total_weight - container.max_weight) ** 2

        # Centre of mass
        cm_x = sum(self.weights[i] * self.positions[i].x for i in range(n)) / total_weight
        cm_y = sum(self.weights[i] * self.positions[i].y for i in range(n)) / total_weight

        if not (0.2 * container.width <= cm_x <= 0.8 * container.width):
            penalties["centre_of_mass"] += abs(cm_x - container.width / 2) ** 2
        if not (0.2 * container.depth <= cm_y <= 0.8 * container.depth):
            penalties["centre_of_mass"] += abs(cm_y - container.depth / 2) ** 2

        # Reward for CM proximity
        dist_cm = math.sqrt(
            (cm_x - container.width / 2) ** 2 +
            (cm_y - container.depth / 2) ** 2
        )
        rewards["centre_mass_reward"] = max(0.0, 100.0 - dist_cm)

        # Cylinder centrality
        max_dist = math.sqrt(
            (container.width / 2) ** 2 +
            (container.depth / 2) ** 2
        )

        for i, pos in enumerate(self.positions):
            d = math.sqrt(
                (pos.x - container.width / 2) ** 2 +
                (pos.y - container.depth / 2) ** 2
            )
            rewards["cylinder_centrality"] += (
                self.weights[i] / max(self.weights)
            ) * (1 - d / max_dist) ** 2 * 50

        # Results
        weighted_penalty = (
            penalties["overlap"] * 10.0 +
            penalties["bounds"] * 5.0 +
            penalties["capacity"] * 100.0 +
            penalties["centre_of_mass"] * 2.0 +
            penalties["loading_order"] * 10.0
        )

        fitness = rewards["centre_mass_reward"] + rewards["cylinder_centrality"] - weighted_penalty
        self.fitness = fitness

        is_successful = all(p == 0 for p in penalties.values())

        return {
            "fitness": fitness,
            "penalties": penalties,
            "rewards": rewards,
            "total_weight": total_weight,
            "centre_of_mass": (cm_x, cm_y),
            "is_successful": is_successful,
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

        # Draw centre of mass and safe zone
        total_weight = sum(self.weights)
        if total_weight > 0:
            cm_x = sum(self.weights[i] * self.positions[i].x for i in range(len(self.weights))) / total_weight
            cm_y = sum(self.weights[i] * self.positions[i].y for i in range(len(self.weights))) / total_weight

            # centre of mass marker
            ax.plot(cm_x, cm_y, "rx", markersize=12, markeredgewidth=2, label="centre of Mass")

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

        # Build title
        if hasattr(self, 'ids'):
            sequence_str = str(self.ids)
            if len(self.ids) > 20:
                # Truncate long sequences
                sequence_str = str(self.ids[:20]) + "..."

            full_title = f"{title}\nFitness: {self.fitness:.2f}\nSequence: {sequence_str}"
        else:
            full_title = f"{title}\nFitness: {self.fitness:.2f}"

        # Set up the plot
        ax.set_xlim(0, container.width)
        ax.set_ylim(0, container.depth)
        ax.set_aspect("equal")
        ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)
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

        self.individuals: List[Individual] = []
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

    def evolve(self, mutation_rate = 0.01, elitism = True):
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

            new_individuals.append(child)

        self.individuals = new_individuals
        self.evaluate_population()

    def get_stats(self):
        """
        Calculates population statistics.

        Returns:
            Dictionary with 'best_fitness', 'avg_fitness', 'worst_fitness', 'num_successful', 'success_rate'
        """

        stats = {
            "best_fitness": -float("inf"),
            "avg_fitness": 0.0,
            "worst_fitness": float("inf"),
            "num_successful": 0,
            "success_rate": 0.0,
        }

        total_fitness = 0.0

        for ind in self.individuals:
            res = ind.evaluate(self.container)
            f = res["fitness"]

            stats["best_fitness"] = max(stats["best_fitness"], f)
            stats["worst_fitness"] = min(stats["worst_fitness"], f)
            total_fitness += f

            if res["is_successful"]:
                stats["num_successful"] += 1

        stats["avg_fitness"] = total_fitness / len(self.individuals)
        stats["success_rate"] = stats["num_successful"] / len(self.individuals)

        return stats


def run_single_instance(instance, mutation_rate=0.01, population_size=200, max_generations=500, print_interval=20, draw_result=True):
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
        Dictionary containing
    """
    # Initialise population
    population = Population(population_size, instance.cylinders, instance.container)
    population.evaluate_population()

    # Evolution
    for gen in range(max_generations):
        population.evolve(mutation_rate)

        if print_interval and gen % print_interval == 0:
            stats = population.get_stats()
            print(f"[Gen {gen:4d}] "
                    f"Best={stats['best_fitness']:.2f} | "
                    f"Avg={stats['avg_fitness']:.2f} | "
                    f"Success={stats['success_rate']*100:.1f}%")

    # Get best solution
    best = population.get_best_individual()
    result = best.evaluate(instance.container)

    print("\nFINAL RESULT")
    print("-" * 50)
    print(f"Fitness: {result['fitness']:.2f}")
    print(f"Successful: {result['is_successful']}")
    print(f"Penalties: {result['penalties']}")

    # Visualise
    if draw_result:
        best.draw(instance.container)

    return result

def run_all_instances(mutation_rate=0.01,population_size=200, max_generations=500, print_interval=50, verbose=True):
    """
    Run the GA on all available instances and analyse results.

    Args:
        mutation_rate: Probability of mutation per gene
        memetic_attempts: Number of local search attempts before stopping
        population_size: Size of population
        max_generations: Maximum number of generations to evolve
        verbose: If True, prints progress during evolution

    Returns:
        Dict with 'results' (Dict as given by evaluate()) and 'success_rate' (float)
    """
    # Gather all instances
    instances = (
        [("Basic", i) for i in container_instances.create_basic_instances()] +
        [("Challenging", i) for i in container_instances.create_challenging_instances()]
    )

    results = []

    for category, instance in instances:
        print("\n" + "=" * 80)
        print(f"{instance.name} ({category})")

        population = Population(population_size, instance.cylinders, instance.container)
        population.evaluate_population()

        for gen in range(max_generations):
            population.evolve(mutation_rate)

            if gen % print_interval == 0:
                stats = population.get_stats()
                print(f"Gen {gen:4d}: "
                        f"Best={stats['best_fitness']:.2f}, "
                        f"Success={stats['success_rate']*100:.1f}%")

        best = population.get_best_individual()
        res = best.evaluate(instance.container)

        print("\nFINAL")
        print(f"Fitness: {res['fitness']:.2f}")
        print(f"Successful: {res['is_successful']}")
        print(f"Penalties: {res['penalties']}")

        results.append({
            "instance": instance.name,
            "category": category,
            "fitness": res["fitness"],
            "successful": res["is_successful"],
            "penalties": res["penalties"],
            "sequence": best.ids,
        })

        best.draw(instance.container,
                    title=f"{instance.name} | Success={res['is_successful']}")

    success_rate = sum(r["successful"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print(f"OVERALL SUCCESS RATE: {success_rate*100:.1f}%")

    return {
        "results": results,
        "success_rate": success_rate
    }

def evaluate_and_visualise_sequence(instance, id_sequence, show_visualization=True):
    """
    Evaluate and visualise a specific sequence of cylinder IDs.

    Args:
        instance: The problem instance (from container_instances)
        id_sequence: List of cylinder IDs in the order to place them
        show_visualization: If True, show the visualization plot

    Returns:
        Dict as given by evaluate()
    """

    # Create a dictionary to map IDs to cylinders
    cyl_map = {c.id: c for c in instance.cylinders}
    # Create individual with given sequence
    individual = Individual([cyl_map[i] for i in id_sequence])

    result = individual.evaluate(instance.container)

    print("\nSEQUENCE EVALUATION")
    print("-" * 60)
    print(f"Sequence: {id_sequence}")
    print(f"Fitness: {result['fitness']:.2f}")
    print(f"Successful: {result['is_successful']}")
    print("Penalties:")
    for k, v in result["penalties"].items():
        print(f"  {k}: {v:.4f}")

    if show_visualization:
        individual.draw(instance.container)

    return result

def main():
    # GA parameters
    mutation_rate = 0.04
    population_size = 200
    max_generations = 200

    # Load instances
    basic_instances: List[Instance] = container_instances.create_basic_instances()
    challenging_instances: List[Instance] = container_instances.create_challenging_instances()


    # --------------------------------------------------------------------------------------
    # You may choose 3 methods of running the genetic algorithm by uncommenting
    # an option and commenting the others.
    #
    # For Option 1 & Option 3, choose an instance to evaluate
    #
    # Option 2 will run all instances. The GA will only continue to the next instance
    # once the plot draw() window has been closed
    #
    # For Option 3, please also input a sequence/solution you would like to evaluate
    #
    # Uncomment an option and execute to run it. (Comment the other options)
    # ---------------------------------------------------------------------------------------------------

    #! Choose an instance to run for option 1 or option 3
    instance = basic_instances[2] # Max = 2
    instance = challenging_instances[3] # Max = 3

    # #! Option 1: Run single given instance
    # run_single_instance(instance=instance, mutation_rate=mutation_rate, population_size=population_size, max_generations=max_generations, print_interval=20, draw_result=True)

    #! Option 2: Run all instances
    run_all_instances( mutation_rate=mutation_rate, population_size=population_size, max_generations=max_generations, verbose=True)

    # #! Option 3: Test specific sequence for chosen instance
    # sequence = [6, 2, 1, 8, 5, 7, 10, 4, 9, 3]  # NOTE: Ensure ids used are present in instance
    # evaluate_and_visualise_sequence(instance, sequence, show_visualization=True)

if __name__ == "__main__":
    main()
