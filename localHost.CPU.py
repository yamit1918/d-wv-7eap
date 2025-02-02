# 7procss
import time
import multiprocessing

def cpu_intensive_task():
    # Function to perform CPU-intensive calculations
    while True:
        [x**2 for x in range(10**6)]

def main():
    # Create multiple processes to maximize CPU usage
    processes = []
    for _ in range(7):  # Running 7 processes to increase CPU usage
        process = multiprocessing.Process(target=cpu_intensive_task)
        processes.append(process)
        process.start()

    # Let the processes run for 30 seconds
    time.sleep(30)

    # Terminate the processes after 30 seconds
    for process in processes:
        process.terminate()
        process.join()

    print("CPU task completed.")

if __name__ == "__main__":
    main()
