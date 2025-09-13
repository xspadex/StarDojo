import os
import json
import re
import time
import argparse
import logging
import uuid
from typing import List, Dict, Any, Tuple, IO, Callable, Optional
import random
import numpy as np
from PIL import Image

# 假設你的環境代碼位於一個名為 stardew_env 的包中
# 如果不是，請相應地修改導入路徑
from eval_agent import StarDojoLLMIsolated
from tasks.farming import Farming

# --- 配置區 ---

# 1. 定義輸出數據的根資料夾
OUTPUT_DIR = "/Users/xspadex/iclr/traj_data/agent_runs/"
IMAGE_SAVE_PATH = "/Users/xspadex/iclr/traj_data/agent_runs/images"
# 3. 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

REPEAT_NUM = 1000

class RuleBasedAgent:
    """
    在星露谷物語環境中，根據一個給定的策略函數 (policy function) 來執行動作。
    策略函數接收當前的觀測(obs)作為輸入，並返回一個動作字串。
    腳本會記錄每一步的 (觀測, 動作, 新觀測)，並將結果聚合輸出到單一的 jsonl 檔案中。
    """
    def __init__(self, port: int, output_dir: str):
        """
        初始化基於規則的代理。

        Args:
            port (int): 遊戲伺服器端口。
            output_dir (str): 保存輸出數據的根目錄。
        """
        self.port = port
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, "images")
        self.env = None
        self.action_proxy = None

        os.makedirs(self.images_dir, exist_ok=True)
        logging.info(f"所有輸出將保存在: {os.path.abspath(self.output_dir)}")
        logging.info(f"所有截圖將保存在: {os.path.abspath(self.images_dir)}")

    def _parse_action(self, action_string: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        """
        解析動作字串，支援位置參數和關鍵字參數。
        (此函數與原腳本保持不變)
        """
        match = re.match(r"(\w+)\((.*)\)", action_string)
        if not match:
            raise ValueError(f"無法解析動作字串: {action_string}")

        name, args_str = match.groups()
        args_str = args_str.strip()
        
        args = []
        kwargs = {}

        if not args_str:
            return name, args, kwargs

        def _capture(*pargs, **pkwargs):
            return pargs, pkwargs

        try:
            captured_args, captured_kwargs = eval(
                f"_capture({args_str})", 
                {"__builtins__": None}, 
                {"_capture": _capture}
            )
            args = list(captured_args)
            kwargs = captured_kwargs
        except Exception as e:
            logging.error(f"使用 eval 解析參數 '{args_str}' 失敗: {e}")
            raise ValueError(f"無法解析參數: {args_str}")

        return name, args, kwargs

    def _make_dict_serializable(self, data: Any) -> Any:
        """遞歸地將字典中非 JSON 可序列化的類型轉換為可序列化類型。(與原腳本保持不變)"""
        if isinstance(data, dict):
            return {k: self._make_dict_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._make_dict_serializable(i) for i in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def _capture_and_save_image(self, obs: Dict[str, Any]) -> str:
        """從觀測數據中提取截圖，保存並返回相對路徑。(與原腳本保持不變)"""
        try:
            rgb_image = obs['screenshot'][:, :, :3].astype(np.uint8)
            img = Image.fromarray(rgb_image)
            image_filename = f"{uuid.uuid4().hex}.jpeg"
            full_image_path = os.path.join(self.images_dir, image_filename)
            img.save(full_image_path, 'JPEG')
        except Exception as e:
            logging.error(f"保存截圖失敗: {full_image_path}. 錯誤: {e}")
            return ""
        return full_image_path
        
    def _prepare_obs_for_json(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """準備觀測數據以便寫入 JSON。(與原腳本保持不變)"""
        obs_data = {k: v for k, v in obs.items() if k != 'screenshot' and k != 'image_paths'}
        return obs_data

    def execute_task(
        self, 
        task_data: Dict[str, Any], 
        policy_function: Callable[[Dict[str, Any], int], str],
        max_steps: int,
        jsonl_writer: IO[str]
    ):
        """
        執行單個任務，在每個步驟中調用策略函數來決定下一個動作。

        Args:
            task_data (Dict[str, Any]): 包含任務設置（如存檔名、初始命令）的字典。
            policy_function (Callable): 一個函數，接收 obs 和 step_count，返回動作字串或 None。
            max_steps (int): 此任務最多執行的步數。
            jsonl_writer (IO[str]): 用於寫入結果的檔案對象。
        """
        task_name = task_data.get("task", f"unnamed_task_{time.strftime('%Y%m%d%H%M%S')}")
        save_name = task_data.get("save")
        task_type = task_data.get("task_type")
        task_id = task_data.get("task_id")
        init_commands = task_data.get("init_commands")

        logging.info(f"--- 開始執行任務: '{task_name}' (存檔: '{save_name}', 初始命令: '{init_commands}') ---")
        
        task_result = {
            "task": task_name,
            "save": save_name,
            "init_commands": init_commands,
            "initial_state": {},
            "steps": []
        }
        
        from env.tasks.utils import load_task
        task_for_launch = load_task.load_task(task_type, task_id)
        completed = False
        try:
            # --- 環境初始化 ---
            self.env = StarDojoLLMIsolated(
                port=self.port,
                new_game=False,
                image_save_path=IMAGE_SAVE_PATH,
                output_video=False,
                max_image_storage=1,
                task=task_for_launch,
            )
            self.action_proxy = self.env.action_proxy
            self.action_proxy.wait_for_server()
            logging.info("環境連接成功。")
            time.sleep(3)
            self.env.action_proxy.exit_menu()
            time.sleep(0.5)
            self.env.action_proxy.choose_option(0)
            time.sleep(0.5)
            self.env.action_proxy.teleport_random()
            time.sleep(0.5)

            # --- 步驟 0: 捕獲並記錄初始狀態 ---
            current_obs = self.env._get_obs()
            initial_image_path = self._capture_and_save_image(current_obs)
            initial_obs_data = self._prepare_obs_for_json(current_obs)
            
            task_result["initial_state"] = {
                "image_path": initial_image_path
            }
            logging.info("已記錄初始狀態。")

            # --- 主循環：根據策略函數執行動作 ---
            for i in range(max_steps):
                # 1. 調用策略函數獲取動作
                action_str = policy_function(task_name, current_obs, i)

                # 2. 檢查終止信號
                if action_str is None:
                    logging.info(f"步驟 {i+1}: 策略函數返回 None，任務提前結束。")
                    break
                
                logging.info(f"步驟 {i+1}/{max_steps}: {action_str}")

                try:
                    # 3. 解析並執行動作
                    action_name, args, kwargs = self._parse_action(action_str)
                    action_func = getattr(self.action_proxy, action_name)
                    action_func(*args, **kwargs)
                    time.sleep(1.0) # 等待動作執行完成

                    # 4. 捕獲並記錄動作後的狀態
                    obs_after_action = self.env._get_obs()
                    image_path_after_action = self._capture_and_save_image(obs_after_action)
                    obs_data_after_action = self._prepare_obs_for_json(obs_after_action)

                    step_data = {
                        "action": action_str,
                        "image_path": image_path_after_action
                    }
                    task_result["steps"].append(step_data)
                    
                    if self.env.task.evaluate(obs_after_action, self.env.task_proxy)['completed']:
                        logging.info(f"任務已完成。")
                        completed = True
                        break

                    # 5. 更新當前觀測，用於下一次循環
                    current_obs = obs_after_action
                
                except (ValueError, AttributeError, TypeError) as e:
                    logging.error(f"處理動作 '{action_str}' 時出錯: {e}")
                    break
            else: # for 循環正常結束後執行
                logging.info(f"已達到最大步數 {max_steps}，任務結束。")
            
            if not completed:
                logging.info(f"任務未完成。")
                task_result["completed"] = False
                return
            else:
                task_result["completed"] = True
            # --- 寫入結果 ---
            json_line = json.dumps(self._make_dict_serializable(task_result), ensure_ascii=False)
            jsonl_writer.write(json_line + '\n')
            logging.info(f"--- 任務 '{task_name}' 處理完畢並已寫入 jsonl 檔案 ---")

        except Exception as e:
            logging.critical(f"在執行任務 '{task_name}' 期間發生嚴重錯誤: {e}", exc_info=True)
        finally:
            if self.env:
                self.env.exit()
                self.env = None
                logging.info("環境連接已關閉。")

    def run_from_file(
        self, 
        input_jsonl_path: str, 
        policy_function: Callable[[str, Dict[str, Any], int], str],
        max_steps: int
    ):
        """
        從指定的 jsonl 檔案讀取任務設置，並使用策略函數執行每個任務。

        Args:
            input_jsonl_path (str): 輸入的 jsonl 檔案路徑（只讀取任務設置，忽略 "actions"）。
            policy_function (Callable): 用於決定每一步動作的函數。
            max_steps (int): 每個任務最多執行的步數。
        """
        input_file_name = os.path.basename(input_jsonl_path).split(".")[0]
        output_jsonl_path = os.path.join(self.output_dir, f"agent_run_data_{input_file_name}.jsonl")
        logging.info(f"準備從 '{input_jsonl_path}' 讀取任務設置...")
        logging.info(f"輸出將寫入到 '{output_jsonl_path}'")
        
        try:
            if os.path.exists(output_jsonl_path):
                # 為了安全起見，避免意外覆蓋，可以選擇引發錯誤或提示用戶
                logging.warning(f"輸出檔案已存在: {output_jsonl_path}. 將會附加到檔案末尾。")
            
            with open(output_jsonl_path, 'a', encoding='utf-8') as writer:
                with open(input_jsonl_path, 'r', encoding='utf-8') as reader:
                    lines = reader.readlines()
                    for i in range(REPEAT_NUM):
                        for line in lines:
                            if line.strip():
                                task_data = json.loads(line)
                                # 注意：這裡我們傳遞了 policy_function 和 max_steps
                                self.execute_task(task_data, policy_function, max_steps, writer)
        except FileNotFoundError:
            logging.critical(f"輸入檔案未找到: {input_jsonl_path}")
        except json.JSONDecodeError as e:
            logging.critical(f"解碼 JSON 時出錯: {e}")
        except Exception as e:
            logging.critical(f"在處理檔案時出錯: {e}", exc_info=True)
        
        logging.info("檔案中的所有任務都已處理完畢。")


def is_path_clear(observation: Dict[str, Any], nearest: Optional[Tuple[Tuple[int, int], float]]) -> bool:
    """
    檢查玩家與 'nearest' 物體之間的直線路徑是否可以通行，包含終點。

    Args:
        observation: 包含 'player' 和 'viewingtiles' 的遊戲狀態字典。
        nearest: find_nearest 函數返回的元組 (相對位置, 距離)，或 None。

    Returns:
        如果路徑上所有中間的地塊以及終點都可通行，則返回 True，否則返回 False。
    """
    # 如果沒有找到目標物體，則直接返回 False
    if nearest is None:
        return False

    # 從 observation 中獲取玩家的絕對座標
    player_pos = tuple(observation['player']['position'])
    
    # 計算目標物體的絕對座標
    target_relative_pos = nearest[0]
    target_abs_pos = (player_pos[0] + target_relative_pos[0], player_pos[1] + target_relative_pos[1])

    # 如果目標就在旁邊（距離小於2），中間沒有任何地塊。
    # 這種情況下，我們仍然需要檢查終點本身是否可通行。
    # 因此，我們讓程式碼繼續往下走，由後面的迴圈來處理單一終點的檢查。
    # if nearest[1] < 2.0:
    #     return True  <-- 移除這個捷徑，以確保終點總是被檢查

    # 為了快速查找，將 viewingtiles 轉換成以座標為鍵 (key) 的字典
    tile_map = {tuple(tile['position']): tile for tile in observation['viewingtiles']}

    # --- 使用 Bresenham's line algorithm 取得路徑上的所有座標點 ---
    path_points: List[Tuple[int, int]] = []
    x0, y0 = player_pos
    x1, y1 = target_abs_pos
    
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        path_points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    # --- 演算法結束 ---

    # 檢查路徑上的所有地塊 (不包含起點，但包含終點)
    # path_points[0] 是玩家位置, path_points[-1] 是目標位置
    if len(path_points) <= 1:
        return True # 目標就是玩家自身位置，路徑為空

    # <<< 修改處 >>>
    # 將迴圈的結束點從 len(path_points) - 1 改為 len(path_points)
    # 這樣就會包含路徑的最後一個點 (終點)
    for i in range(1, len(path_points)):
        pos = path_points[i]
        tile = tile_map.get(pos)

        # 如果路徑上的某個地塊不在視野範圍內，為安全起見，視為不可通行
        if tile is None:
            return False

        # 根據您定義的規則檢查地塊是否可通行
        object_at_tile = tile.get('object_at_tile', '').lower()
        terrain_at_tile = tile.get('terrain_at_tile', '').lower()

        # 條件1: 地塊上的物體必須是空
        object_is_passable = (object_at_tile == "")
        
        # 條件2: 地塊的地形必須是 'grass' 或空
        terrain_is_passable = (terrain_at_tile.endswith("grass") or terrain_at_tile == "")

        # 兩個條件必須同時滿足，否則路徑被阻擋
        if not (object_is_passable and terrain_is_passable):
            return False

    # 如果迴圈順利跑完，代表路徑上所有中間地塊及終點都可通行
    return True

def find_nearest(observation: Dict[str, Any], object_name: str) -> Optional[Tuple[Tuple[int, int], float]]:
    """
    找到最近的物體。
    """
    nearest = None
    player_position = observation['player']['position']
    if object_name.lower() == "tree" or object_name.lower() == "grass":
        for tile in observation['viewingtiles']:
            tile_position = tile['position']
            terrain_at_tile = tile['terrain_at_tile']
            if terrain_at_tile.lower().endswith(object_name.lower()):
                distance = np.linalg.norm(np.array(tile_position) - np.array(player_position))
                if nearest is None or (distance < nearest[1] and distance > 0):
                    nearest = ([tile_position[0]-player_position[0], tile_position[1]-player_position[1]], distance)
    elif object_name.lower() in ["weeds", "stone", "twig"]:
        for tile in observation['viewingtiles']:
            tile_position = tile['position']
            object_at_tile = tile['object_at_tile']
            if object_at_tile.lower().endswith(object_name.lower()):
                distance = np.linalg.norm(np.array(tile_position) - np.array(player_position))
                if nearest is None or (distance < nearest[1] and distance > 0):
                    nearest = ([tile_position[0]-player_position[0], tile_position[1]-player_position[1]], distance)
    elif object_name.lower() == "dirt":
        for tile in observation['viewingtiles']:
            tile_position = tile['position']
            tile_properties = tile['tile_properties']
            terrain_at_tile = tile['terrain_at_tile']
            object_at_tile = tile['object_at_tile']
            if "dirt" in tile_properties.lower() and not terrain_at_tile.lower().endswith("hoedirt") and not terrain_at_tile.lower().endswith("tree") and len(object_at_tile) == 0:
                distance = np.linalg.norm(np.array(tile_position) - np.array(player_position))
                if nearest is None or (distance < nearest[1] and distance > 0):
                    nearest = ([tile_position[0]-player_position[0], tile_position[1]-player_position[1]], distance)
    return nearest

def chop_wood_policy(observation: Dict[str, Any]) -> str:
    current_item = observation['player']['currentinventory']['currentitem']
    needs_change_item = current_item.lower() != "axe"
    axe_index = 0
    inventory = observation['inventory']
    for i, item in enumerate(inventory):
        if ("Name" in item and item["Name"].lower() == "axe") or ("name" in item and item["name"].lower() == "axe"):
            axe_index = i
            break
    nearest_tree = find_nearest(observation, "tree")
    nearest_twig = find_nearest(observation, "twig")
    
    if nearest_tree is not None and nearest_tree[1] == 1:
        if nearest_tree[0][0] == 1 and nearest_tree[0][1] == 0:
            return f"use(direction='right')"
        elif nearest_tree[0][0] == -1 and nearest_tree[0][1] == 0:
            return f"use(direction='left')"
        elif nearest_tree[0][0] == 0 and nearest_tree[0][1] == 1:
            return f"use(direction='down')"
        else:
            return f"use(direction='up')"
    elif nearest_twig is not None and nearest_twig[1] == 1:
        if nearest_twig[0][0] == 1 and nearest_twig[0][1] == 0:
            return f"use(direction='right')"
        elif nearest_twig[0][0] == -1 and nearest_twig[0][1] == 0:
            return f"use(direction='left')"
        elif nearest_twig[0][0] == 0 and nearest_twig[0][1] == 1:
            return f"use(direction='down')"
        else:
            return f"use(direction='up')"
    elif nearest_tree is None and nearest_twig is None:
        random_x = random.randint(-5, 5)
        random_y = random.randint(-5, 5)
        return f"move(x={random_x}, y={random_y})"
    
    twig_left_accessible = nearest_twig is not None and is_path_clear(observation, ((nearest_twig[0][0]-1, nearest_twig[0][1]), nearest_twig[1]-1))
    if twig_left_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_twig[0][0]-1}, y={nearest_twig[0][1]})"
    twig_right_accessible = nearest_twig is not None and is_path_clear(observation, ((nearest_twig[0][0]+1, nearest_twig[0][1]), nearest_twig[1]-1))
    if twig_right_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_twig[0][0]+1}, y={nearest_twig[0][1]})"
    twig_up_accessible = nearest_twig is not None and is_path_clear(observation, ((nearest_twig[0][0], nearest_twig[0][1]-1), nearest_twig[1]-1))
    if twig_up_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_twig[0][0]}, y={nearest_twig[0][1]-1})"
    twig_down_accessible = nearest_twig is not None and is_path_clear(observation, ((nearest_twig[0][0], nearest_twig[0][1]+1), nearest_twig[1]-1))
    if twig_down_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_twig[0][0]}, y={nearest_twig[0][1]+1})"
    
    tree_left_accessible = nearest_tree is not None and is_path_clear(observation, ((nearest_tree[0][0]-1, nearest_tree[0][1]), nearest_tree[1]-1))
    if tree_left_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_tree[0][0]-1}, y={nearest_tree[0][1]})"
    tree_right_accessible = nearest_tree is not None and is_path_clear(observation, ((nearest_tree[0][0]+1, nearest_tree[0][1]), nearest_tree[1]-1))
    if tree_right_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_tree[0][0]+1}, y={nearest_tree[0][1]})"
    tree_up_accessible = nearest_tree is not None and is_path_clear(observation, ((nearest_tree[0][0], nearest_tree[0][1]-1), nearest_tree[1]-1))
    if tree_up_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_tree[0][0]}, y={nearest_tree[0][1]-1})"
    tree_down_accessible = nearest_tree is not None and is_path_clear(observation, ((nearest_tree[0][0], nearest_tree[0][1]+1), nearest_tree[1]-1))
    if tree_down_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={axe_index})"
        return f"move(x={nearest_tree[0][0]}, y={nearest_tree[0][1]+1})"
    return clear_weeds_policy(observation)

def cut_grass_policy(observation: Dict[str, Any]) -> str:
    current_item = observation['player']['currentinventory']['currentitem']
    needs_change_item = current_item.lower() != "scythe"
    scythe_index = 0
    inventory = observation['inventory']
    for i, item in enumerate(inventory):
        if ("Name" in item and item["Name"].lower() == "scythe") or ("name" in item and item["name"].lower() == "scythe"):
            scythe_index = i
            break
    
    nearest_grass = find_nearest(observation, "grass")
    if nearest_grass is not None and nearest_grass[1] == 1:
        if nearest_grass[0][0] == 1 and nearest_grass[0][1] == 0:
            return f"use(direction='right')"
        elif nearest_grass[0][0] == -1 and nearest_grass[0][1] == 0:
            return f"use(direction='left')"
        elif nearest_grass[0][0] == 0 and nearest_grass[0][1] == 1:
            return f"use(direction='down')"
        else:
            return f"use(direction='up')"
    elif nearest_grass is None:
        random_x = random.randint(-5, 5)
        random_y = random.randint(-5, 5)
        return f"move(x={random_x}, y={random_y})"
    
    grass_left_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0]-1, nearest_grass[0][1]), nearest_grass[1]-1))
    if grass_left_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]-1}, y={nearest_grass[0][1]})"
    grass_right_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0]+1, nearest_grass[0][1]), nearest_grass[1]-1))
    if grass_right_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]+1}, y={nearest_grass[0][1]})"
    grass_up_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0], nearest_grass[0][1]-1), nearest_grass[1]-1))
    if grass_up_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]}, y={nearest_grass[0][1]-1})"
    grass_down_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0], nearest_grass[0][1]+1), nearest_grass[1]-1))
    if grass_down_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]}, y={nearest_grass[0][1]+1})"
    
    return clear_weeds_policy(observation)

def get_item_quantity(observation: Dict[str, Any], item_name: str) -> int:
    """
    獲取物品的數量。
    """
    inventory = observation['inventory']
    count = 0
    for item in inventory:
        if "Name" in item and item['Name'] is not None and item['Name'].lower() == item_name.lower():
            if "Quantity" in item:
                count += item['Quantity']
            else:
                count += item['quantity']
        elif "name" in item and item['name'] is not None and item['name'].lower() == item_name.lower():
            if "Quantity" in item:
                count += item['Quantity']
            else:
                count += item['quantity']
    return count

def clear_weeds_policy(observation: Dict[str, Any]) -> str:
    current_item = observation['player']['currentinventory']['currentitem']
    needs_change_item = current_item.lower() != "scythe"
    scythe_index = 0
    inventory = observation['inventory']
    for i, item in enumerate(inventory):
        if ("Name" in item and item["Name"].lower() == "scythe") or ("name" in item and item["name"].lower() == "scythe"):
            scythe_index = i
            break
    nearest_grass = find_nearest(observation, "weeds")
    if nearest_grass is not None and nearest_grass[1] == 1:
        if nearest_grass[0][0] == 1 and nearest_grass[0][1] == 0:
            return f"use(direction='right')"
        elif nearest_grass[0][0] == -1 and nearest_grass[0][1] == 0:
            return f"use(direction='left')"
        elif nearest_grass[0][0] == 0 and nearest_grass[0][1] == 1:
            return f"use(direction='down')"
        else:
            return f"use(direction='up')"
    elif nearest_grass is None:
        random_x = random.randint(-5, 5)
        random_y = random.randint(-5, 5)
        return f"move(x={random_x}, y={random_y})"
    
    grass_left_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0]-1, nearest_grass[0][1]), nearest_grass[1]-1))
    if grass_left_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]-1}, y={nearest_grass[0][1]})"
    grass_right_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0]+1, nearest_grass[0][1]), nearest_grass[1]-1))
    if grass_right_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]+1}, y={nearest_grass[0][1]})"
    grass_up_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0], nearest_grass[0][1]-1), nearest_grass[1]-1))
    if grass_up_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]}, y={nearest_grass[0][1]-1})"
    grass_down_accessible = nearest_grass is not None and is_path_clear(observation, ((nearest_grass[0][0], nearest_grass[0][1]+1), nearest_grass[1]-1))
    if grass_down_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={scythe_index})"
        return f"move(x={nearest_grass[0][0]}, y={nearest_grass[0][1]+1})"
    return clear_stone_policy(observation)
    
def clear_stone_policy(observation: Dict[str, Any]) -> str:
    current_item = observation['player']['currentinventory']['currentitem']
    
    needs_change_item = current_item.lower() != "pickaxe"
    pickaxe_index = 0
    inventory = observation['inventory']
    for i, item in enumerate(inventory):
        if ("Name" in item and item["Name"].lower() == "pickaxe") or ("name" in item and item["name"].lower() == "pickaxe"):
            pickaxe_index = i
            break
    
    nearest_stone = find_nearest(observation, "stone")
    if nearest_stone is not None and nearest_stone[1] == 1:
        if nearest_stone[0][0] == 1 and nearest_stone[0][1] == 0:
            return f"use(direction='right')"
        elif nearest_stone[0][0] == -1 and nearest_stone[0][1] == 0:
            return f"use(direction='left')"
        elif nearest_stone[0][0] == 0 and nearest_stone[0][1] == 1:
            return f"use(direction='down')"
        else:
            return f"use(direction='up')"
    elif nearest_stone is None:
        random_x = random.randint(-5, 5)
        random_y = random.randint(-5, 5)
        return f"move(x={random_x}, y={random_y})"
    stone_left_accessible = nearest_stone is not None and is_path_clear(observation, ((nearest_stone[0][0]-1, nearest_stone[0][1]), nearest_stone[1]-1))
    if stone_left_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={pickaxe_index})"
        return f"move(x={nearest_stone[0][0]-1}, y={nearest_stone[0][1]})"
    stone_right_accessible = nearest_stone is not None and is_path_clear(observation, ((nearest_stone[0][0]+1, nearest_stone[0][1]), nearest_stone[1]-1))
    if stone_right_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={pickaxe_index})"
        return f"move(x={nearest_stone[0][0]+1}, y={nearest_stone[0][1]})"
    stone_up_accessible = nearest_stone is not None and is_path_clear(observation, ((nearest_stone[0][0], nearest_stone[0][1]-1), nearest_stone[1]-1))
    if stone_up_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={pickaxe_index})"
        return f"move(x={nearest_stone[0][0]}, y={nearest_stone[0][1]-1})"
    stone_down_accessible = nearest_stone is not None and is_path_clear(observation, ((nearest_stone[0][0], nearest_stone[0][1]+1), nearest_stone[1]-1))
    if stone_down_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={pickaxe_index})"
        return f"move(x={nearest_stone[0][0]}, y={nearest_stone[0][1]+1})"
    return chop_wood_policy(observation)
    
def till_tile_policy(observation: Dict[str, Any]) -> str:
    current_item = observation['player']['currentinventory']['currentitem']
    
    needs_change_item = current_item.lower() != "hoe"
    hoe_index = 0
    inventory = observation['inventory']
    for i, item in enumerate(inventory):
        if ("Name" in item and item["Name"].lower() == "hoe") or ("name" in item and item["name"].lower() == "hoe"):
            hoe_index = i
            break
    
    nearest_dirt = find_nearest(observation, "dirt")
    if nearest_dirt is not None and nearest_dirt[1] == 1:
        if nearest_dirt[0][0] == 1 and nearest_dirt[0][1] == 0:
            return f"use(direction='right')"
        elif nearest_dirt[0][0] == -1 and nearest_dirt[0][1] == 0:
            return f"use(direction='left')"
        elif nearest_dirt[0][0] == 0 and nearest_dirt[0][1] == 1:
            return f"use(direction='down')"
        else:
            return f"use(direction='up')"
    elif nearest_dirt is None:
        random_x = random.randint(-5, 5)
        random_y = random.randint(-5, 5)
        return f"move(x={random_x}, y={random_y})"
    
    dirt_left_accessible = nearest_dirt is not None and is_path_clear(observation, ((nearest_dirt[0][0]-1, nearest_dirt[0][1]), nearest_dirt[1]-1))
    if dirt_left_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={hoe_index})"
        return f"move(x={nearest_dirt[0][0]-1}, y={nearest_dirt[0][1]})"
    dirt_right_accessible = nearest_dirt is not None and is_path_clear(observation, ((nearest_dirt[0][0]+1, nearest_dirt[0][1]), nearest_dirt[1]-1))
    if dirt_right_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={hoe_index})"
        return f"move(x={nearest_dirt[0][0]+1}, y={nearest_dirt[0][1]})"
    dirt_up_accessible = nearest_dirt is not None and is_path_clear(observation, ((nearest_dirt[0][0], nearest_dirt[0][1]-1), nearest_dirt[1]-1))
    if dirt_up_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={hoe_index})"
        return f"move(x={nearest_dirt[0][0]}, y={nearest_dirt[0][1]-1})"
    dirt_down_accessible = nearest_dirt is not None and is_path_clear(observation, ((nearest_dirt[0][0], nearest_dirt[0][1]+1), nearest_dirt[1]-1))
    if dirt_down_accessible:
        if needs_change_item:
            return f"choose_item(slot_index={hoe_index})"
        return f"move(x={nearest_dirt[0][0]}, y={nearest_dirt[0][1]+1})"
    return chop_wood_policy(observation)

# ==============================================================================
# --- 您需要實現的部分：策略函數 (Policy Function) ---
# ==============================================================================
def my_policy_function(task_name: str, observation: Dict[str, Any], step_count: int) -> str or None:
    """
    這是一個策略函數的範例。
    您可以基於 'observation' 中的任何資訊來決定下一步的動作。

    Args:
        task_name (str): 任務名稱。
        observation (Dict[str, Any]): 環境的當前觀測，包含 'player_status', 'game_time' 等。
        step_count (int): 當前的步數 (從 0 開始)。

    Returns:
        str: 一個表示動作的字串，例如 "move(x=1)"。
        None: 如果返回 None，則表示任務終止。
    """
    # 範例邏輯：向前移動5步，然後向右移動5步，然後停止。
    if task_name == "chop_10_wood_with_axe":
        # for tile in observation['viewingtiles']:
        #     tile_position = tile['position']
        #     if not tile["debris_at_tile"].endswith("388"):
        #         continue
        #     else:
        #         return f"move(x={tile_position[0]-observation['player']['position'][0]}, y={tile_position[1]-observation['player']['position'][1]})"
        return chop_wood_policy(observation)
    elif task_name == "craft_1_wood_fence":
        if get_item_quantity(observation, "Wood") >= 2:
            return 'craft(item="Wood Fence")'
        else:
            return chop_wood_policy(observation)
    elif task_name == "forage_10_hay_with_scythe":
        return cut_grass_policy(observation)
    elif task_name == "clear_10_weeds_with_scythe":
        return clear_weeds_policy(observation)
    elif task_name == "clear_5_stone_with_pickaxe":
        return clear_stone_policy(observation)
    elif task_name == "till_5_tile_with_hoe":
        return till_tile_policy(observation)
    else:
        # 10步後，返回 None 來結束任務
        logging.info("策略決定結束任務。")
        return "move(x=1, y=1)"

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在星露谷物語中根據策略函數執行動作，並將結果輸出到 JSONL 檔案。")
    parser.add_argument("--input_file", type=str, default="/Users/xspadex/Cursors/official/StarDojo/env/rule_traj_tasks.jsonl", help="包含任務設置的輸入 JSONL 檔案路徑。")
    parser.add_argument("--port", type=int, default=10783, help="遊戲伺服器的端口號。")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="保存輸出數據（jsonl 和 images）的根目錄。")
    parser.add_argument("--max_steps", type=int, default=30, help="每個任務最多執行的步數。")
    
    args = parser.parse_args()

    # 創建代理實例
    agent = RuleBasedAgent(
        port=args.port, 
        output_dir=args.output_dir
    )
    
    try:
        # 執行任務，傳入我們定義的策略函數
        agent.run_from_file(
            input_jsonl_path=args.input_file,
            policy_function=my_policy_function,
            max_steps=args.max_steps
        )
    except KeyboardInterrupt:
        logging.info("接收到用戶中斷信號，正在退出...")
    finally:
        logging.info("程序已退出。")