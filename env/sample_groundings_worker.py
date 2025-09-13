import os
import json
import uuid
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

# Assume your environment code is in a package named stardew_env
# If not, please modify the import path accordingly
from stardew_env import StarDojo
from eval_agent import StarDojoLLMIsolated

class DataCollector:
    """
    A class for automatically collecting VLM grounding data from the Stardew Valley environment.
    Designed to be run as an isolated worker process.
    """
    def __init__(self, port: int, save_index: int, output_dir: str, repeat_num: int):
        """
        Initializes the data collector.

        Args:
            port (int): The game server port.
            save_index (int): The game save index.
            output_dir (str): The root directory for this worker's output data.
            repeat_num (int): The number of times to teleport and collect data on each map.
        """
        self.output_path = output_dir
        self.screenshots_path = os.path.join(self.output_path, "screenshots")
        self.metadata_path = os.path.join(self.output_path, "metadata.jsonl")
        self.repeat_num = repeat_num
        
        # Create output directories for this specific worker
        os.makedirs(self.screenshots_path, exist_ok=True)
        
        logging.info(f"[Worker-{port}] Connecting to Stardew Valley environment...")
        from env.tasks.utils import load_task
        task = load_task.load_task("farming_lite", 5)
        
        self.env = StarDojoLLMIsolated(
            port=port,
            save_index=save_index,
            new_game=True,
            task=task,
            image_save_path=self.screenshots_path,
            output_video=False,
            max_image_storage=1
        )
        
        
        self.action_proxy = self.env.action_proxy
        self.metadata = []
        
        self.action_proxy.wait_for_server()
        logging.info(f"[Worker-{port}] Environment connection successful!")

    def get_last_part(self, s: str) -> str:
        """Extracts the last part from a 'Type.Subtype.Name' formatted string."""
        if isinstance(s, str) and '.' in s:
            return s.split('.')[-1]
        return s

    def extract_objects_from_tile(self, tile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts all relevant item information from a single tile's data.
        """
        objects = []
        position = tile.get('position')
        if not position:
            return []

        object_keys = {
            'debris_at_tile',
            'object_at_tile',
            # 'crop_at_tile',
            'terrain_at_tile',
            'door_info'
        }

        for key in object_keys:
            obj_data = tile.get(key)
            if obj_data:
                obj_name = ""
                if isinstance(obj_data, str):
                    obj_name = self.get_last_part(obj_data)
                elif key == 'door_info' and isinstance(obj_data, str):
                    obj_name = 'exit to ' + obj_data
                
                if obj_name:
                    objects.append({'name': obj_name, 'position': position})
        
        return objects

    def scan_current_map_and_collect_data(self):
        """
        Scans the current visible area, generates, and saves data for each discovered item.
        """
        current_location = self.env._get_obs().get('location', 'Unknown Location')
        logging.info(f"Scanning map: {current_location}")
        
        try:
            obs = self.env._get_obs()
        except Exception as e:
            logging.error(f"Error getting observation data: {e}")
            return

        tiles_to_scan = obs.get('viewingtiles')
        if tiles_to_scan is None:
            logging.warning("'viewingtiles' not found, falling back to 'surroundingsdata'.")
            tiles_to_scan = obs.get('surroundingsdata', [])

        player_pos = obs.get('player', {}).get('position')
        
        if not tiles_to_scan or not player_pos:
            logging.warning("Incomplete observation data (missing tiles or player position), skipping scan.")
            return

        rgb_image = obs['screenshot'][:, :, :3].astype(np.uint8)
        image_filename = f"{uuid.uuid4().hex}.jpeg"
        full_image_path = os.path.join(self.screenshots_path, image_filename)
        
        try:
            img = Image.fromarray(rgb_image)
            img.save(full_image_path, 'JPEG')
        except Exception as e:
            logging.error(f"Failed to save screenshot: {full_image_path}. Error: {e}")
            return

        collected_count = 0
        crops = obs.get('crops', [])
        for crop in crops:
            crop_name = crop['id']
            position = crop['position']
            if abs(position[0]) > 9 or abs(position[1]) > 5:
                continue
            current_phase = crop['current_phase']
            data_record = {
                "image_file": os.path.join("screenshots", image_filename), # Use relative path
                "object_name": crop_name,
                "object_position": position,
                "player_position": player_pos,
                "map_name": current_location,
                "type": "crop",
            }
            with open(self.metadata_path, 'a') as f:
                    f.write(json.dumps(data_record) + '\n')
                
            collected_count += 1
        
        for tile in tiles_to_scan:
            objects_on_tile = self.extract_objects_from_tile(tile)
            
            for obj in objects_on_tile:
                data_record = {
                    "image_file": os.path.join("screenshots", image_filename), # Use relative path
                    "object_name": obj['name'],
                    "object_position": obj['position'],
                    "player_position": player_pos,
                    "map_name": current_location
                }
                
                with open(self.metadata_path, 'a') as f:
                    f.write(json.dumps(data_record) + '\n')
                
                collected_count += 1
            

        
        logging.info(f"Scan complete, collected {collected_count} new data points in this area.")

    def run_collection(self, maps_to_visit: List[str]):
        """
        Executes the full data collection flow for a given list of maps.
        
        Args:
            maps_to_visit (List[str]): The list of map names this worker is responsible for.
        """
        for map_name in maps_to_visit:
            logging.info(f"--- Traveling to map: {map_name} ---")
            try:
                self.action_proxy.warp(map_name, 10, 10)
                time.sleep(3) # Wait for the scene to load
            except Exception as e:
                logging.error(f"An error occurred while warping to {map_name}: {e}")
                continue

            for i in range(self.repeat_num):
                logging.info(f"Performing random collection {i+1}/{self.repeat_num} on {map_name}...")
                self.action_proxy.teleport_random() 
                time.sleep(0.5) # Wait for screen content to update
                import random
                direction = random.choice([0,1,2,3])
                self.action_proxy.turn(direction)
                time.sleep(0.1)
                self.scan_current_map_and_collect_data()
        
        self.cleanup()

    def cleanup(self):
        """
        Cleans up resources and exits.
        """
        logging.info("Data collection for this worker is complete. Closing environment connection...")
        self.env.exit()
        logging.info("Program has exited.")

def run_collection_task(port: int, save_index: int, output_dir: str, repeat_num: int, maps_for_worker: List[str], log_path: str):
    """
    The main entry point function for a single collection process.
    
    Args:
        port (int): The unique port for this process.
        save_index (int): The unique save index for this process.
        output_dir (str): The unique output directory for this process.
        repeat_num (int): The number of collection iterations per map.
        maps_for_worker (List[str]): The subset of maps this process will handle.
        log_path (str): The path to the log file for this process.
    """
    # Configure logging for this specific process
    log_formatter = logging.Formatter(
        '%(asctime)s - P%(process)d - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    
    # Get the root logger and add the file handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Avoid adding handlers multiple times if the logger is already configured
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logging.info(f"Worker process started. Assigned maps: {maps_for_worker}")
    
    collector = None
    try:
        collector = DataCollector(
            port=port, 
            save_index=save_index,
            output_dir=output_dir,
            repeat_num=repeat_num
        )
        collector.run_collection(maps_to_visit=maps_for_worker)
    except KeyboardInterrupt:
        logging.info("User interruption signal received.")
    except Exception as e:
        logging.error(f"An unhandled exception occurred in the worker: {e}", exc_info=True)
    finally:
        if collector:
            collector.cleanup()
        logging.info("Worker process finished.")