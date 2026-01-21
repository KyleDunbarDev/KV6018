from main import Vector2, Individual, Population
import container_instances
from container_instances import Cylinder, Container, Instance
import random
import math

def main():
    random.seed(42)

    #! Pick instance
    # instance = container_instances.create_basic_instances()[2] # Max = 2
    instance = container_instances.create_challenging_instances()[2] # Max = 3
    compare_placement_strategies(instance)


def evaluate_random_placement(instance, show_visualisation=True):
    """
    Randomly places cylinders inside the container (ignoring constraints),
    evaluates penalties and fitness, and prints results to console.

    Args:
        instance: Instance to be evaluated upon

        show_visualisation: (optional) draws plot if true
    """

    container = instance.container
    cylinders = instance.cylinders

    individual = Individual(cylinders.copy())

    individual.positions = []
    radii = [c.diameter / 2 for c in cylinders]

    # Random placement
    for r in radii:
        x = random.uniform(r, container.width - r)
        y = random.uniform(r, container.depth - r)
        individual.positions.append(Vector2(x, y))

    # Evaluate using unified logic
    result = individual.evaluate(container)

    # Print results
    print("\n" + "=" * 80)
    print(f"RANDOM PLACEMENT EVALUATION — {instance.name}")
    print("=" * 80)

    print(f"Fitness: {result['fitness']:.2f}")
    print(f"Successful (strict feasibility): {result['is_successful']}")

    print("\nPenalties:")
    for k, v in result["penalties"].items():
        print(f"  {k:<16}: {v:.4f}")

    print("\nRewards:")
    for k, v in result["rewards"].items():
        print(f"  {k:<16}: {v:.4f}")

    cm_x, cm_y = result["centre_of_mass"]
    print(f"\nCentre of Mass: ({cm_x:.2f}, {cm_y:.2f})")
    print(f"Total Weight:   {result['total_weight']:.2f} / {container.max_weight:.2f}")

    print("\nCylinder Positions:")
    print(f"{'ID':<4} {'X':<8} {'Y':<8} {'Radius':<8}")
    print("-" * 40)
    for cyl, pos, r in zip(cylinders, individual.positions, radii):
        print(f"{cyl.id:<4} {pos.x:<8.2f} {pos.y:<8.2f} {r:<8.2f}")

    if show_visualisation:
        individual.draw(
            container,
            title=f"Random Placement | Fitness={result['fitness']:.2f}"
        )

    return result

def evaluate_greedy_placement(instance, samples_per_cylinder=40,
                              show_visualisation=True):
    """
    Greedy placement:
    - Cylinders placed in given order
    - For each cylinder, sample candidate positions
    - Choose position with lowest penalty

    Args:
        instance: Instance to be evaluated upon

        show_visualisation: (optional) draws plot if true
    """

    container = instance.container
    cylinders = instance.cylinders

    individual = Individual(cylinders.copy())
    radii = [c.diameter / 2 for c in cylinders]
    individual.positions = []

    # func to calculate penalties for candidate positions
    def calc_penalty(pos, idx):
        r_i = radii[idx]
        penalty = 0.0

        # Boundary
        if pos.x - r_i < 0:
            penalty += (r_i - pos.x) ** 2
        if pos.x + r_i > container.width:
            penalty += (pos.x + r_i - container.width) ** 2
        if pos.y - r_i < 0:
            penalty += (r_i - pos.y) ** 2
        if pos.y + r_i > container.depth:
            penalty += (pos.y + r_i - container.depth) ** 2

        # Interactions with already placed cylinders
        for j, prev in enumerate(individual.positions):
            r_j = radii[j]
            dx = pos.x - prev.x
            dy = pos.y - prev.y
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = r_i + r_j

            # Overlap
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2

            # Loading order (j < idx by construction)
            if prev.y > pos.y and abs(dx) < min_dist:
                vertical_gap = prev.y - pos.y
                if vertical_gap < min_dist:
                    penalty += (min_dist - vertical_gap) ** 2

        return penalty

    # Greedy placement
    for i, r in enumerate(radii):
        best_pos = None
        best_penalty = float("inf")

        for _ in range(samples_per_cylinder):
            pos = Vector2(
                random.uniform(r, container.width - r),
                random.uniform(r, container.depth - r)
            )

            p = calc_penalty(pos, i)

            if p < best_penalty:
                best_penalty = p
                best_pos = pos

        # Guaranteed placement
        if best_pos is None:
            best_pos = Vector2(
                random.uniform(r, container.width - r),
                random.uniform(r, container.depth - r)
            )

        individual.positions.append(best_pos)

    # Evaluate final result
    result = individual.evaluate(container)

    # Console output
    print("\n" + "=" * 80)
    print(f"GREEDY PLACEMENT — {instance.name}")
    print("=" * 80)
    print(f"Fitness: {result['fitness']:.2f}")
    print(f"Successful: {result['is_successful']}")
    print("Penalties:", result["penalties"])

    if show_visualisation:
        individual.draw(
            container,
            title=f"Greedy Placement | Fitness={result['fitness']:.2f}"
        )

    return result

def evaluate_ga_solution(instance, mutation_rate=0.04, population_size=200, max_generations=200, show_visualisation=True):
    """
    Runs the GA for given instance, simpler returns for comparison

    Args:
        instance: Instance to be evaluated upon
        mutation_rate: (optional) Chance of sequence swap per gene
        population_size: (optional) Size of population per generation
        max_generations: (optional) When to stop GA
        show_visualisation:  (optional) draw plot
    """
    print("\n" + "=" * 80)
    print(f"Running GA...")
    # Create population
    population = Population(population_size, instance.cylinders, instance.container)
    population.evaluate_population()

    # Evolve
    for _ in range(max_generations):
        population.evolve(mutation_rate)

    # Get best
    best = population.get_best_individual()
    result = best.evaluate(instance.container)

    if show_visualisation:
        best.draw(
            instance.container,
            title=f"GA Solution | Fitness={result['fitness']:.2f}"
        )
    return result


def compare_placement_strategies(instance):
    """
    Compare Random, Greedy, and GA-based placement on the same instance.

    Args:
        instance: Instance to have algorithms tested upon
    """
    random.seed(42)

    print("\n" + "=" * 100)
    print(f"PLACEMENT STRATEGY COMPARISON — {instance.name}")
    print("=" * 100)

    random_res = evaluate_random_placement(instance, show_visualisation=False)

    greedy_res = evaluate_greedy_placement(instance, show_visualisation=False)

    ga_res = evaluate_ga_solution(instance, show_visualisation=False)

    rows = [
        ("Random", random_res),
        ("Greedy", greedy_res),
        ("GA (Memetic)", ga_res),
    ]

    print("\n" + "-" * 100)
    print(f"{'Method':<14} {'Fitness':>10} {'Success':>10} "
          f"{'p_Overlap':>10} {'p_Bounds':>10} {'p_CM':>10} {'p_Order':>10} {'r_CM':>10} {'r_Central':>10}")
    print("-" * 100)

    for name, res in rows:
        p = res["penalties"]
        r = res["rewards"]
        print(f"{name:<14} "
              f"{res['fitness']:>10.2f} "
              f"{str(res['is_successful']):>10} "
              f"{p['overlap']:>10.2f} "
              f"{p['bounds']:>10.2f} "
              f"{p['centre_of_mass']:>10.2f} "
              f"{p['loading_order']:>10.2f}"
              f"{r['centre_mass_reward']:>10.2f}"
              f"{r['cylinder_centrality']:>10.2f}")

    print("-" * 100)

    return {
        "random": random_res,
        "greedy": greedy_res,
        "ga": ga_res,
    }

if __name__ == "__main__":
    main()
