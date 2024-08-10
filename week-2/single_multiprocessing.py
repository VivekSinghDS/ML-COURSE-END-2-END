import time
import random
from multiprocessing import Pool

def inference_task(data):
    time.sleep(random.uniform(0.1, 0.3))
    return data * 2

def run_single_process(data_list):
    start_time = time.time()
    results = [inference_task(data) for data in data_list]
    end_time = time.time()
    return results, end_time - start_time

def run_multi_process(data_list, num_processes):
    start_time = time.time()
    with Pool(num_processes) as pool:
        results = pool.map(inference_task, data_list)
    end_time = time.time()
    return results, end_time - start_time

def main():
    data_list = list(range(100))
    
    single_results, single_time = run_single_process(data_list)
    print(f"Single-process time: {single_time:.4f} seconds")
    
    for num_processes in [2, 4, 8]:
        multi_results, multi_time = run_multi_process(data_list, num_processes)
        print(f"{num_processes}-process time: {multi_time:.4f} seconds")
        speedup = single_time / multi_time
        print(f"Speedup: {speedup:.2f}x")
        print()

if __name__ == "__main__":
    main()