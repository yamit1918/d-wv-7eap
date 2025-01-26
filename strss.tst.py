import time
import dimod
import asyncio
import logging
import matplotlib.pyplot as plt
from dwave.system import EmbeddingComposite, DWaveSampler

# Setup logging
logging.basicConfig(filename='quantum_task.log', level=logging.INFO)

# Define function to create a complex BQM problem
def create_bqm(num_variables):
    linear = {f'v{i}': 1 for i in range(num_variables)}
    quadratic = {(f'v{i}', f'v{(i+1) % num_variables}'): -1 for i in range(num_variables)}
    return dimod.BinaryQuadraticModel(linear, quadratic, offset=0.0, vartype=dimod.BINARY)

# Asynchronous function to submit and process a quantum task
async def quantum_task(sampler, num_variables, duration=3600):
    bqm = create_bqm(num_variables)
    start_time = time.time()
    results = []
    while time.time() - start_time < duration:
        response = await sampler.sample_async(bqm)
        for sample, energy in response.data(['sample', 'energy']):
            results.append((energy, sample))
            logging.info(f'Sample: {sample}, Energy: {energy}')
    return results

# Main asynchronous function to run multiple quantum tasks in parallel
async def main():
    sampler = EmbeddingComposite(DWaveSampler())
    tasks = []
    num_tasks = 10  # Increase the number of parallel tasks
    
    # Create and run tasks with varying complexities
    for i in range(num_tasks):
        num_variables = (i + 1) * 6  # Increase complexity dynamically
        task = quantum_task(sampler, num_variables)
        tasks.append(task)
    
    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # For logging and analysis
    energies = [result[0][0] for result in results if result]
    
    # Plot results
    plt.plot(range(len(energies)), energies, label='Energy')
    plt.xlabel('Task Index')
    plt.ylabel('Energy')
    plt.title('Quantum Task Energy Analysis')
    plt.legend()
    plt.show()

# Run the main function
asyncio.run(main())
