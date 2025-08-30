import multiprocessing
import os
import time
from typing import List, Dict, Any

# Assume the function run_stardojo_isolated is in a file named 'main_runner.py'
# If your file structure is different, you'll need to adjust the import.
from eval_agent import run_stardojo_isolated

# --- Configuration ---

# 1. Define the list of tasks you want to run in parallel.
# Each task is a dictionary specifying its 'name' and 'id'.
TASKS_TO_RUN: List[Dict[str, Any]] = [
    {"name": "farming_lite", "id": 0},
    # {"name": "farming_lite", "id": 1},
    # {"name": "meet_villagers", "id": 1},
    # {"name": "catch_a_fish", "id": 0},
    # {"name": "plant_parsnips", "id": 2},
    # Add more tasks as needed
]

# 2. Set the starting port and save index. 
# Each process will get an incremented value to ensure they are isolated.
BASE_PORT = 10783
BASE_SAVE_INDEX = 0

# 3. Define base paths for logs and image outputs.
# The script will create unique subdirectories for each process.
BASE_LOG_PATH = "../env/parallel_logs"
BASE_IMAGE_SAVE_PATH = "../env/parallel_images"

# 4. Other common settings for all processes.
NEW_GAME_FOR_ALL = True
OUTPUT_VIDEO = False
NEEDS_SHARED_MEMORY = False

# --- End of Configuration ---


def main():
    """
    Main function to orchestrate the creation and management of parallel processes.
    """
    print("--- Stardojo Multiprocess Manager ---")

    # Create base directories if they don't exist
    os.makedirs(BASE_LOG_PATH, exist_ok=True)
    os.makedirs(BASE_IMAGE_SAVE_PATH, exist_ok=True)

    processes = []
    
    # Loop through the configured tasks and set up a process for each one
    for i, task_info in enumerate(TASKS_TO_RUN):
        task_name = task_info["name"]
        task_id = task_info["id"]

        # --- Create unique parameters for this specific process ---
        
        # Assign a unique port and save index
        process_port = BASE_PORT + i
        process_save_index = BASE_SAVE_INDEX + i
        
        # Create a unique identifier for this run, e.g., "buy_tools_0"
        process_identifier = f"{task_name}_{task_id}"

        # Create unique paths for logs and images
        log_dir = os.path.join(BASE_LOG_PATH, process_identifier)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"run_{process_port}.log")

        image_save_path = os.path.join(BASE_IMAGE_SAVE_PATH, process_identifier)
        os.makedirs(image_save_path, exist_ok=True)

        print(f"Preparing Process {i+1}/{len(TASKS_TO_RUN)}:")
        print(f"  - Task: {process_identifier}")
        print(f"  - Port: {process_port}")
        print(f"  - Save Index: {process_save_index}")
        print(f"  - Log File: {log_path}\n")

        # --- Create the Process object ---
        
        # Using kwargs is cleaner for functions with many arguments
        process_args = {
            "port": process_port,
            "save_index": process_save_index,
            "new_game": NEW_GAME_FOR_ALL,
            "image_save_path": image_save_path,
            "output_video": OUTPUT_VIDEO,
            "task_name": task_name,
            "task_id": task_id,
            "log_path": log_path,
            "needs_shared_memory": NEEDS_SHARED_MEMORY,
        }

        p = multiprocessing.Process(
            target=run_stardojo_isolated,
            kwargs=process_args
        )
        
        processes.append(p)

    # --- Start and Manage Processes ---

    # Start all the processes
    print(f"--- Starting {len(processes)} processes in parallel. This may take a moment. ---")
    for p in processes:
        p.start()
        # A small delay can prevent resource spikes and make logs easier to follow
        time.sleep(5) 

    # Wait for all processes to complete
    # The .join() method blocks the main script until the target process finishes.
    print("\n--- All processes are running. Waiting for them to complete. ---")
    print("You can monitor their progress by checking the log files in the 'parallel_logs' directory.")
    
    for p in processes:
        p.join()

    print("\n--- All tasks have been completed. ---")


if __name__ == "__main__":
    # The `if __name__ == "__main__":` block is essential for multiprocessing.
    # It ensures that child processes do not re-execute the process creation code.
    main()