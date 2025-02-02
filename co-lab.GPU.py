# @title GPU - in colab if u wont change run enviroment to GPU, ull get error
import torch
import time
import os
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.multiprocessing import Process

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

def complex_model_training(size=128, iterations=20, batch_size=32):
    """Tests training with a complex model."""
    print(f"Testing complex model training with size {size}, iterations {iterations}, batch size {batch_size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        class ComplexModel(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.fc = nn.Linear(128 * (size // 4) * (size // 4), size)

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = ComplexModel(size).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Dummy data: generate random input tensors with batch dimension
        inputs = torch.rand(batch_size, 3, size, size, device=device) # BxCxHxW
        targets = torch.rand(batch_size, size, device=device) # BxS

        start_time = time.time()
        for _ in range(iterations):
            optimizer.zero_grad() # Zeroes previous gradients
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, targets) # calculate loss
            loss.backward() # calculate gradients
            optimizer.step() # apply update to the weights

        end_time = time.time()

        print(f"Complex model training took {end_time - start_time:.4f} seconds")
        return end_time-start_time
    except Exception as e:
        print(f"Error during complex model training: {e}")
        return None

def distributed_training(size=128, iterations=10):
    """Tests distributed training with a simple model."""
    print(f"Testing distributed training with size {size}, iterations {iterations}")
    world_size = 2
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        processes = []
        for rank in range(world_size):
          p = Process(target=run_distributed_training, args=(rank, world_size, size, iterations))
          p.start()
          processes.append(p)
        for p in processes:
            p.join()
        return True
    except Exception as e:
        print(f"Error during distributed training: {e}")
        return False

def run_distributed_training(rank, world_size, size, iterations):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
      model = nn.Linear(size,size).to(device)
      model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

      optimizer = Adam(model.parameters(), lr=0.01)
      criterion = nn.MSELoss()

      x = torch.rand(10, size, device=device)
      y = torch.rand(10, size, device=device)

      for _ in range(iterations):
          optimizer.zero_grad()
          outputs = model(x)
          loss = criterion(outputs, y)
          loss.backward()
          optimizer.step()
      dist.destroy_process_group()
    except Exception as e:
        print(f"Error during distributed training: {e}")
    return True

def run_tests(size_mult=1):
    """Runs all tests."""
    results = []
    matrix_size = int(1024*size_mult)
    conv_size = int(256*size_mult)
    channels = int(128*size_mult)
    train_size = int(128*size_mult)
    iterations=int(20/size_mult)
    batch_size=int(32/size_mult)

    mult_time = large_matrix_multiplication(matrix_size)
    results.append({"test": "matrix multiplication", "size": matrix_size, "time":mult_time})

    conv_time = large_convolution(channels,conv_size)
    results.append({"test":"convolution", "channels": channels, "size": conv_size, "time":conv_time})

    train_time = complex_model_training(train_size,iterations,batch_size)
    results.append({"test":"complex model training", "size": train_size, "iterations": iterations,"batch_size": batch_size, "time":train_time})

    dist_time = distributed_training(train_size,int(iterations/2))
    results.append({"test":"distributed training", "size": train_size, "iterations": int(iterations/2),"time":dist_time})

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
