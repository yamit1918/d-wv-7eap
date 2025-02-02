import torch
import time
import numpy as np
import torch.multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_operation(size, iterations):
    """Performs a CPU-intensive calculation using a tensor."""
    a = torch.rand(size, size) # create a random CPU tensor

    start_time = time.time()
    # Perform a sequence of CPU-heavy operations.
    for _ in range(iterations):
      a = torch.cos(a) # cosine
      a = torch.sqrt(torch.abs(a)) # absolute value and square root
      a = torch.log10(torch.abs(a)) # absolute value and log10
      a = torch.sin(a) # sine
      torch.matmul(a,a) # Matrix multiplication, even if small size

    end_time = time.time()
    return end_time - start_time

def memory_intensive_operation(size, num_copies):
    """Performs memory copy operations"""
    a = torch.rand(size,size)
    start_time = time.time()
    for _ in range(num_copies):
        b = a.clone()
    end_time = time.time()
    return end_time-start_time

def worker_function(size, iterations, num_copies, use_memory):
    """Worker function that calls a cpu bound operation repeatedly"""
    total_time = 0
    if use_memory:
      total_time += memory_intensive_operation(size,num_copies)
    total_time += cpu_bound_operation(size,iterations)

    return total_time

def stress_cpu_multiprocessing(size, num_processes, iterations, use_memory, num_copies):
  """Uses multiprocessing to stress the CPU cores."""
  print(f"Testing multiprocessing CPU with tensors of size ({size}, {size})")
  start_time = time.time()

  with ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = executor.map(worker_function, [size]*num_processes, [iterations]*num_processes,[num_copies]*num_processes, [use_memory]*num_processes )
    total_time = sum(results)

  end_time = time.time()
  print(f"Multiprocessing CPU operation took {end_time-start_time:.4f} seconds. Total computation time in all processes: {total_time}")
  return end_time-start_time

def stress_cpu_recursion(size, depth, use_memory):
    """Stress CPU with a recursive computation"""
    if depth <= 0:
      return cpu_bound_operation(size,1)
    else:
      return stress_cpu_recursion(size, depth-1, use_memory) + cpu_bound_operation(size,1)

def run_tests(size_mult=1):
    """Runs all tests."""
    results = []
    size = int(128*size_mult)
    iterations=int(1000/size_mult)
    num_processes = os.cpu_count() # check number of cores available
    use_memory = True
    num_copies = int(1000*size_mult)
    depth = int(10*size_mult)

    cpu_time = stress_cpu_multiprocessing(size, num_processes, iterations, use_memory, num_copies)
    results.append({"test": "multiprocessing CPU with memory", "size": size, "num_processes":num_processes, "iterations":iterations, "num_copies":num_copies, "time": cpu_time})

    cpu_time_recursion = stress_cpu_recursion(size, depth, False)
    results.append({"test": "recursive CPU", "size": size, "depth":depth,"time":cpu_time_recursion})

    return results

if __name__ == '__main__':
    # sizes is multiplied by size_mult to have larger stress tests.
    # Start with small numbers, then increase
    size_multipliers = [1, 2, 4, 8, 16]
    all_results = []
    for size_mult in size_multipliers:
      print("------------------------------------------------------------")
      print(f"Running tests with size multiplier: {size_mult}")
      results = run_tests(size_mult=size_mult)
      all_results.append({"size_multiplier": size_mult, "results": results})

    import json
    with open('cpu_stress_test_results.json', 'w') as f:
      json.dump(all_results, f, indent=4)

    print("CPU stress test complete. Results saved to cpu_stress_test_results.json")
