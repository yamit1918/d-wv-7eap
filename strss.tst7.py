import time
import dimod
import asyncio
from dwave.system import EmbeddingComposite, DWaveSampler

# Define function to create a complex BQM problem
def create_bqm(num_variables):
    linear = {f'v{i}': 1 for i in range(num_variables)}
    quadratic = {(f'v{i}', f'v{(i+1) % num_variables}'): -1 for i in range(num_variables)}
    return dimod.BinaryQuadraticModel(linear, quadratic, offset=0.0, vartype=dimod.BINARY)

# Asynchronous function to submit and process a quantum task
async def quantum_task(sampler, bqm, duration=30):
    start_time = time.time()
    while time.time() - start_time < duration:
        response = await sampler.sample_async(bqm)
        for sample, energy in response.data(['sample', 'energy']):
            print(f'Sample: {sample}, Energy: {energy}')

# Main asynchronous function to run multiple quantum tasks in parallel
async def main():
    sampler = EmbeddingComposite(DWaveSampler())
    tasks = []
    num_tasks = 4  # Adjust the number of parallel tasks

    # Create BQM problems with different complexities
    for i in range(num_tasks):
        bqm = create_bqm(num_variables=8)  # Increase this number for more complexity
        task = quantum_task(sampler, bqm)
        tasks.append(task)

    # Run tasks concurrently
    await asyncio.gather(*tasks)

# Run the main function
asyncio.run(main())
