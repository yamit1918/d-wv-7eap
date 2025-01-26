import time
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite

def create_problem(num_variables, density):
    """Creates a random QUBO (Quadratic Unconstrained Binary Optimization) problem."""
    num_interactions = int(density * num_variables * (num_variables - 1) / 2)
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    variables = list(range(num_variables))

    import random
    interactions = random.sample([(i, j) for i in variables for j in variables if i < j], num_interactions)
    for u, v in interactions:
        bqm.add_interaction(u, v, random.uniform(-1, 1))
    for v in variables:
        bqm.add_variable(v, random.uniform(-1, 1))
    return bqm

def run_annealing(bqm, num_reads=1000):
    """Runs annealing on the D-Wave system."""
    sampler = EmbeddingComposite(DWaveSampler())
    start_time = time.time()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    end_time = time.time()
    return sampleset, end_time - start_time

def test_dwave(start_vars = 10, end_vars = 100, step = 10, density = 0.5, num_reads = 1000):
    """Tests D-Wave with increasing problem size."""
    print("Starting D-Wave test...")
    print(f"Testing from {start_vars} to {end_vars} variables in steps of {step}, density {density}, {num_reads} reads.")
    results = []
    for num_variables in range(start_vars, end_vars + 1, step):
        print(f"Creating problem with {num_variables} variables...")
        bqm = create_problem(num_variables, density)

        print(f"Running annealing with {num_variables} variables...")
        sampleset, elapsed_time = run_annealing(bqm, num_reads)

        energy = sampleset.first.energy

        results.append({
            "num_variables": num_variables,
            "elapsed_time": elapsed_time,
            "best_energy": energy,
            "num_reads": num_reads
        })
        print(f"{num_variables} variables done in {elapsed_time:.2f} seconds. Best energy: {energy}")

    return results

if __name__ == "__main__":
    test_results = test_dwave()

    # Process results (e.g., plot data, save to file)
    import json
    with open('dwave_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)

    print("D-Wave test complete. Results saved to dwave_test_results.json")
