import torch
import time
import numpy as np

def large_matrix_multiplication(size=1024):
    """Tests matrix multiplication with large tensors."""
    print(f"Testing matrix multiplication with tensors of size ({size}, {size})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)

        start_time = time.time()
        c = torch.matmul(a, b)
        end_time = time.time()

        print(f"Matrix multiplication took {end_time - start_time:.4f} seconds")
        return end_time-start_time
    except Exception as e:
        print(f"Error during matrix multiplication: {e}")
        return None

def large_convolution(channels=128, size=256):
    """Tests convolutions with large tensors."""
    print(f"Testing convolution with tensors of size ({channels}, {size}, {size})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        input_tensor = torch.rand(1, channels, size, size, device=device) # batched tensors
        conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1).to(device) # convolution object

        start_time = time.time()
        output_tensor = conv(input_tensor)
        end_time = time.time()

        print(f"Convolution took {end_time - start_time:.4f} seconds")
        return end_time-start_time
    except Exception as e:
        print(f"Error during convolution: {e}")
        return None


def large_model_training(size=1024, iterations=20):
  """Tests training with a basic linear model on large datasets"""
  print(f"Testing model training with size {size} for {iterations} iterations")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  try:
    model = torch.nn.Linear(size, size).to(device) # basic linear model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Adam optimizer
    criterion = torch.nn.MSELoss() # Mean squared error loss function

    x = torch.rand(10,size, device=device) # input data
    y = torch.rand(10, size, device=device) # output data

    start_time = time.time()
    for _ in range(iterations):
        optimizer.zero_grad() # zero gradients
        outputs = model(x) # forward pass
        loss = criterion(outputs, y) # calculate loss
        loss.backward() # calculate the backward pass of the net
        optimizer.step() # optimize
    end_time=time.time()

    print(f"Training took {end_time - start_time:.4f} seconds")
    return end_time-start_time

  except Exception as e:
        print(f"Error during training: {e}")
        return None

def run_tests(size_mult=1):
    """Runs all tests."""
    results = []
    matrix_size = int(1024*size_mult)
    conv_size = int(256*size_mult)
    channels = int(128*size_mult)
    train_size = int(1024*size_mult)
    iterations=int(20/size_mult)

    mult_time = large_matrix_multiplication(matrix_size)
    results.append({"test": "matrix multiplication", "size": matrix_size, "time":mult_time})

    conv_time = large_convolution(channels,conv_size)
    results.append({"test":"convolution", "channels": channels, "size": conv_size, "time":conv_time})

    train_time = large_model_training(train_size,iterations)
    results.append({"test":"model training", "size": train_size, "iterations": iterations, "time":train_time})

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
    with open('pytorch_stress_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    print("PyTorch stress test complete. Results saved to pytorch_stress_test_results.json")
