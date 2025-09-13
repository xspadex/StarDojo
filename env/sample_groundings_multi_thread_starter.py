import multiprocessing
import os
import time
import numpy as np
from typing import List

# Import the worker function from your refactored data collector script
# Ensure these files are in the same directory or your PYTHONPATH
from sample_groundings_worker import run_collection_task
from eval_agent import StarDojoLLMIsolated

# --- Configuration ---

# 1. Define the complete list of maps you want to collect data from.
#    EACH process will iterate through this ENTIRE list.
ALL_MAPS = [
    "Farm",
    # Add other maps here if needed, e.g., "Town", "Mountain", etc.
]

# 2. Define how many parallel processes you want to run.
#    This should generally not exceed the number of available CPU cores.
NUM_PROCESSES = 3

# 3. Set the starting port and save index. 
#    Each process will get an incremented value to ensure they are isolated.
BASE_PORT = 10783
BASE_SAVE_INDEX = 0

# 4. Define the base output directory for all collected data.
#    The script will create unique subdirectories for each process inside this folder.
BASE_OUTPUT_DIR = "./parallel_vlm_grounding_data"
BASE_LOG_DIR = "./parallel_logs"

# 5. Define the number of random collections to perform on each map.
REPEAT_NUM_PER_MAP = 5000

# --- End of Configuration ---

def main():
    """
    Main function to orchestrate the creation and management of parallel collection processes.
    Each process will run the collection task on the full list of maps.
    """
    print("--- Stardew Valley Multi-Process Data Collector ---")
    print(f"Mode: Each of the {NUM_PROCESSES} processes will sample all maps in the list.")

    # Create base directories if they don't exist
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    processes = []

    # Loop through the number of processes to create, rather than map chunks.
    for i in range(NUM_PROCESSES):
        process_port = BASE_PORT + i
        process_save_index = BASE_SAVE_INDEX + i
        
        # Create a unique identifier and directory for this process's output
        process_identifier = f"worker_{i}_port_{process_port}"
        process_output_dir = os.path.join(BASE_OUTPUT_DIR, process_identifier)
        os.makedirs(process_output_dir, exist_ok=True)

        # Create a unique log file for this process
        log_path = os.path.join(BASE_LOG_DIR, f"{process_identifier}.log")

        # **KEY CHANGE**: Instead of assigning a chunk of maps, we assign the full list to each worker.
        assigned_maps_list = ALL_MAPS

        print(f"\nPreparing Process {i+1}/{NUM_PROCESSES}:")
        print(f"  - Port: {process_port}")
        print(f"  - Save Index: {process_save_index}")
        print(f"  - Maps to Process: {assigned_maps_list}") # Each process gets the full list
        print(f"  - Output Directory: {process_output_dir}")
        print(f"  - Log File: {log_path}")

        # Using kwargs is cleaner for functions with many arguments
        process_args = {
            "port": process_port,
            "save_index": process_save_index,
            "output_dir": process_output_dir,
            "repeat_num": REPEAT_NUM_PER_MAP,
            "maps_for_worker": assigned_maps_list, # Pass the full list here
            "log_path": log_path,
        }

        p = multiprocessing.Process(
            target=run_collection_task,
            kwargs=process_args
        )
        
        processes.append(p)

    # --- Start and Manage Processes ---
    if not processes:
        print("\nNo processes to start. Check your NUM_PROCESSES configuration.")
        return

    print(f"\n--- Starting {len(processes)} processes in parallel. This may take a moment. ---")
    for p in processes:
        p.start()
        # A small delay can prevent resource spikes and make logs easier to follow
        time.sleep(5) 

    print("\n--- All processes are running. Waiting for them to complete. ---")
    print(f"You can monitor their progress by checking the log files in the '{BASE_LOG_DIR}' directory.")
    
    for p in processes:
        p.join() # This blocks the main script until the process finishes.

    print("\n--- All data collection tasks have been completed. ---")

if __name__ == "__main__":
    # The `if __name__ == "__main__":` block is essential for multiprocessing.
    # It ensures that child processes do not re-execute the process creation code.
    main()