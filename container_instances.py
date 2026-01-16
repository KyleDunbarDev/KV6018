"""
KV6018 Cargo Container Loading - Reference Instances
Generate test cases for the container loading problem
"""

import json
from typing import List, Dict

class Cylinder:
    """Represents a cylindrical container"""
    def __init__(self, id: int, diameter: float, weight: float):
        self.id = id
        self.diameter = diameter
        self.weight = weight
    
    def to_dict(self):
        return {
            "id": self.id,
            "diameter": self.diameter,
            "weight": self.weight
        }

class Container:
    """Represents the cargo container"""
    def __init__(self, width: float, depth: float, max_weight: float):
        self.width = width
        self.depth = depth
        self.max_weight = max_weight
    
    def to_dict(self):
        return {
            "width": self.width,
            "depth": self.depth,
            "max_weight": self.max_weight
        }

class Instance:
    """A complete problem instance"""
    def __init__(self, name: str, container: Container, cylinders: List[Cylinder]):
        self.name = name
        self.container = container
        self.cylinders = cylinders
    
    def to_dict(self):
        return {
            "name": self.name,
            "container": self.container.to_dict(),
            "cylinders": [c.to_dict() for c in self.cylinders]
        }
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

# ============================================================================
# BASIC REFERENCE INSTANCES 
# ============================================================================

def create_basic_instances():
    """Simple instances that should be solvable with basic EA"""
    instances = []
    
    # Instance 1: Very simple - 3 identical small cylinders
    inst1 = Instance(
        name="basic_01_three_identical",
        container=Container(10.0, 10.0, 100.0),
        cylinders=[
            Cylinder(1, 2.0, 10.0),
            Cylinder(2, 2.0, 10.0),
            Cylinder(3, 2.0, 10.0)
        ]
    )
    instances.append(inst1)
    
    # Instance 2: Simple - 4 cylinders, two sizes
    inst2 = Instance(
        name="basic_02_two_sizes",
        container=Container(12.0, 10.0, 150.0),
        cylinders=[
            Cylinder(1, 3.0, 20.0),
            Cylinder(2, 3.0, 20.0),
            Cylinder(3, 2.0, 15.0),
            Cylinder(4, 2.0, 15.0)
        ]
    )
    instances.append(inst2)
    
    # Instance 3: 5 cylinders with varied sizes
    inst3 = Instance(
        name="basic_03_varied_sizes",
        container=Container(15.0, 12.0, 200.0),
        cylinders=[
            Cylinder(1, 3.5, 25.0),
            Cylinder(2, 3.0, 20.0),
            Cylinder(3, 2.5, 18.0),
            Cylinder(4, 2.5, 18.0),
            Cylinder(5, 2.0, 15.0)
        ]
    )
    instances.append(inst3)
    
    return instances

# ============================================================================
# CHALLENGING INSTANCES
# ============================================================================

def create_challenging_instances():
    """More difficult instances requiring good optimization"""
    instances = []
    
    # Instance 4: Tight packing required
    inst4 = Instance(
        name="challenge_01_tight_packing",
        container=Container(15.0, 15.0, 300.0),
        cylinders=[
            Cylinder(1, 4.0, 35.0),
            Cylinder(2, 3.5, 30.0),
            Cylinder(3, 3.5, 30.0),
            Cylinder(4, 3.0, 25.0),
            Cylinder(5, 3.0, 25.0),
            Cylinder(6, 2.5, 20.0),
            Cylinder(7, 2.5, 20.0),
            Cylinder(8, 2.0, 15.0)
        ]
    )
    instances.append(inst4)
    
    # Instance 5: Weight distribution challenge (heavy vs light)
    inst5 = Instance(
        name="challenge_02_weight_balance",
        container=Container(18.0, 14.0, 400.0),
        cylinders=[
            Cylinder(1, 3.0, 80.0),  # Heavy
            Cylinder(2, 3.0, 80.0),  # Heavy
            Cylinder(3, 2.5, 10.0),  # Light
            Cylinder(4, 2.5, 10.0),  # Light
            Cylinder(5, 2.5, 10.0),  # Light
            Cylinder(6, 2.5, 10.0),  # Light
            Cylinder(7, 3.5, 60.0),  # Medium-heavy
            Cylinder(8, 3.5, 60.0),  # Medium-heavy
        ]
    )
    instances.append(inst5)
    
    # Instance 6: Many small cylinders
    inst6 = Instance(
        name="challenge_03_many_small",
        container=Container(20.0, 15.0, 350.0),
        cylinders=[
            Cylinder(i, 2.0, 15.0) for i in range(1, 13)
        ]
    )
    instances.append(inst6)
    
    # Instance 7: Mixed sizes with constraint pressure
    inst7 = Instance(
        name="challenge_04_mixed_constraints",
        container=Container(20.0, 20.0, 500.0),
        cylinders=[
            Cylinder(1, 5.0, 50.0),
            Cylinder(2, 4.5, 45.0),
            Cylinder(3, 4.0, 40.0),
            Cylinder(4, 3.5, 35.0),
            Cylinder(5, 3.5, 35.0),
            Cylinder(6, 3.0, 30.0),
            Cylinder(7, 3.0, 30.0),
            Cylinder(8, 2.5, 25.0),
            Cylinder(9, 2.5, 25.0),
            Cylinder(10, 2.0, 20.0)
        ]
    )
    instances.append(inst7)
    
    return instances

# ============================================================================
# GENERATE ALL INSTANCES
# ============================================================================

def generate_all_instances():
    """Generate and save all reference instances"""
    all_instances = {
        "basic": create_basic_instances(),
        "challenging": create_challenging_instances()
    }
    
    # Print summary
    print("=" * 70)
    print("KV6018 CARGO CONTAINER LOADING - REFERENCE INSTANCES")
    print("=" * 70)
    print()
    
    print("BASIC INSTANCES")
    print("-" * 70)
    for inst in all_instances["basic"]:
        print(f"\n{inst.name}:")
        print(f"  Container: {inst.container.width}m × {inst.container.depth}m, "
              f"max weight: {inst.container.max_weight}kg")
        print(f"  Cylinders: {len(inst.cylinders)}")
        total_weight = sum(c.weight for c in inst.cylinders)
        print(f"  Total weight: {total_weight}kg")
    
    print("\n" + "=" * 70)
    print("CHALLENGING INSTANCES")
    print("-" * 70)
    for inst in all_instances["challenging"]:
        print(f"\n{inst.name}:")
        print(f"  Container: {inst.container.width}m × {inst.container.depth}m, "
              f"max weight: {inst.container.max_weight}kg")
        print(f"  Cylinders: {len(inst.cylinders)}")
        total_weight = sum(c.weight for c in inst.cylinders)
        print(f"  Total weight: {total_weight}kg")
    
    print("\n" + "=" * 70)
    print("\nJSON OUTPUT:")
    print("=" * 70)
    
    # Output as JSON
    output = {
        "basic_instances": [inst.to_dict() for inst in all_instances["basic"]],
        "challenging_instances": [inst.to_dict() for inst in all_instances["challenging"]]
    }
    
    print(json.dumps(output, indent=2))
    
    return output

if __name__ == "__main__":
    generate_all_instances()
