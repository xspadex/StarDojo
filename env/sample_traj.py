import os
import json
import re
import time
import argparse
import logging
import uuid
from typing import List, Dict, Any, Tuple, IO

import numpy as np
from PIL import Image

# 假設你的環境代碼位於一個名為 stardew_env 的包中
# 如果不是，請相應地修改導入路徑
from llm_env import StarDojoLLM
from tasks.farming import Farming
# --- 配置區 ---

SKIP_COUNT = 24

# 1. 定義輸出數據的根資料夾
OUTPUT_DIR = "/Users/xspadex/iclr/traj_data/replayed/"
IMAGE_SAVE_PATH = "/Users/xspadex/iclr/traj_data/replayed/images"
# 3. 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ActionReplayer:
    """
    用於從一個定義好的動作序列 jsonl 檔案中，
    在星露谷物語環境中回放動作，並將結果聚合輸出到單一的 jsonl 檔案中。
    """
    def __init__(self, port: int, output_dir: str):
        """
        初始化動作回放器。

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

    # --- MODIFICATION START: Rewritten _parse_action method ---
    def _parse_action(self, action_string: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        """
        解析動作字串，支援位置參數和關鍵字參數。
        例如:
        - 'move(x=1, y=-1)' -> ('move', [], {'x': 1, 'y': -1})
        - 'interact("down")' -> ('interact', ['down'], {})
        - 'tool()' -> ('tool', [], {})
        """
        match = re.match(r"(\w+)\((.*)\)", action_string)
        if not match:
            raise ValueError(f"無法解析動作字串: {action_string}")

        name, args_str = match.groups()
        args_str = args_str.strip()
        
        args = []
        kwargs = {}

        if not args_str:
            # 處理沒有參數的情況, e.g., tool()
            return name, args, kwargs

        # 使用 eval 的一個安全技巧來解析參數
        # 我們定義一個假的函數 `_capture` 來捕獲傳入的 *args 和 **kwargs
        def _capture(*pargs, **pkwargs):
            return pargs, pkwargs

        try:
            # 在一個受限的環境中執行 eval，只允許調用 _capture
            # 這會將 action_string 中的參數傳遞給 _capture
            captured_args, captured_kwargs = eval(
                f"_capture({args_str})", 
                {"__builtins__": None}, 
                {"_capture": _capture}
            )
            args = list(captured_args)
            kwargs = captured_kwargs
        except Exception as e:
            logging.error(f"使用 eval 解析參數 '{args_str}' 失敗: {e}")
            # 作為備用方案，可以添加更簡單的基於正則的解析，但 eval 的方法更通用
            raise ValueError(f"無法解析參數: {args_str}")

        return name, args, kwargs
    # --- MODIFICATION END ---

    def _make_dict_serializable(self, data: Any) -> Any:
        """遞歸地將字典中非 JSON 可序列化的類型 (如 numpy 陣列) 轉換為可序列化類型。"""
        if isinstance(data, dict):
            return {k: self._make_dict_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._make_dict_serializable(i) for i in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def _capture_and_save_image(self, obs: Dict[str, Any]) -> str:
        """
        從觀測數據中提取截圖，保存到中心化的 images 資料夾，並返回相對路徑。

        Args:
            obs (Dict[str, Any]): 從環境中獲取的觀測數據。

        Returns:
            str: 保存後的圖片相對路徑 (e.g., 'images/some_uuid.jpeg')。
        """
        try:
            rgb_image = obs['screenshot'][:, :, :3].astype(np.uint8) # RGB, NOT A
            img = Image.fromarray(rgb_image)
            image_filename = f"{uuid.uuid4().hex}.jpeg"
            full_image_path = os.path.join(self.images_dir, image_filename)
            img.save(full_image_path, 'JPEG')
        except Exception as e:
            logging.error(f"保存截圖失敗: {full_image_path}. 錯誤: {e}")
            return ""
        return full_image_path
        

    def _prepare_obs_for_json(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs_data = {k: v for k, v in obs.items() if k != 'screenshot' and k != 'image_paths'}
        return obs_data

    def replay_task(self, task_data: Dict[str, Any], jsonl_writer: IO[str]):
        """
        執行單個任務的回放，並將結果作為一行寫入到指定的 jsonl 檔案寫入器中。

        Args:
            task_data (Dict[str, Any]): 從 jsonl 檔案中讀取的一行任務數據。
            jsonl_writer (IO[str]): 用於寫入結果的檔案對象。
        """
        task_name = task_data.get("task", f"unnamed_task_{time.strftime('%Y%m%d%H%M%S')}")
        actions = task_data.get("actions", [])
        save_name = task_data.get("save")
        init_commands = task_data.get("init_commands")

        logging.info(f"--- 開始執行任務: '{task_name}' (存檔: '{save_name}', 初始命令: '{init_commands}') ---")
        
        task_result = {
            "task": task_name,
            "save": save_name,
            "init_commands": init_commands,
            "initial_state": {},
            "steps": []
        }
        
        task_for_launch = Farming(task_name, "", 0, "", save_name, init_commands, "", "")

        try:
            self.env = StarDojoLLM(
                port=self.port,
                new_game=False,
                image_save_path=IMAGE_SAVE_PATH,
                output_video=False,
                max_image_storage=1,
            )
            self.action_proxy = self.env.action_proxy
            self.action_proxy.wait_for_server()
            logging.info("環境連接成功。")
            time.sleep(3)
            self.env.action_proxy.exit_to_title()
            time.sleep(1)
            task_for_launch.init_task(self.env.task_proxy)
            time.sleep(1)

            # 步驟 0: 捕獲並記錄初始狀態
            initial_obs = self.env._get_obs()
            initial_image_path = self._capture_and_save_image(initial_obs)
            initial_obs_data = self._prepare_obs_for_json(initial_obs)
            
            task_result["initial_state"] = {
                "observation": initial_obs_data,
                "image_path": initial_image_path
            }
            logging.info("已記錄初始狀態。")

            # 依次執行動作並記錄每一步
            for i, action_str in enumerate(actions):
                logging.info(f"步驟 {i+1}/{len(actions)}: {action_str}")
                try:
                    # --- MODIFICATION START: Update how _parse_action is called and used ---
                    action_name, args, kwargs = self._parse_action(action_str)
                    action_func = getattr(self.action_proxy, action_name)
                    action_func(*args, **kwargs) # Use *args and **kwargs to call the function
                    # --- MODIFICATION END ---
                    time.sleep(1.0)

                    obs_after_action = self.env._get_obs()
                    image_path_after_action = self._capture_and_save_image(obs_after_action)
                    obs_data_after_action = self._prepare_obs_for_json(obs_after_action)

                    step_data = {
                        "action": action_str,
                        "observation": obs_data_after_action,
                        "image_path": image_path_after_action
                    }
                    task_result["steps"].append(step_data)
                
                except (ValueError, AttributeError, TypeError) as e:
                    logging.error(f"處理動作 '{action_str}' 時出錯: {e}")
                    break
            
            # 將整個任務的結果寫入檔案
            json_line = json.dumps(self._make_dict_serializable(task_result), ensure_ascii=False)
            jsonl_writer.write(json_line + '\n')
            logging.info(f"--- 任務 '{task_name}' 處理完畢並已寫入 jsonl 檔案 ---")

        except Exception as e:
            logging.critical(f"在執行任務 '{task_name}' 期間發生嚴重錯誤: {e}")
        finally:
            if self.env:
                self.env.exit()
                self.env = None
                logging.info("環境連接已關閉。")

    def run_from_file(self, input_jsonl_path: str):
        """
        從指定的 jsonl 檔案讀取所有任務並依次執行。

        Args:
            input_jsonl_path (str): 輸入的 jsonl 檔案路徑。
        """
        input_file_name = input_jsonl_path.split("/")[-1].split(".")[0]
        output_jsonl_path = os.path.join(self.output_dir, f"replay_data_{input_file_name}.jsonl")
        logging.info(f"準備從 '{input_jsonl_path}' 讀取任務...")
        logging.info(f"輸出將附加到 '{output_jsonl_path}'")
        
        try:
            if os.path.exists(output_jsonl_path):
                raise FileExistsError(f"輸出檔案已存在: {output_jsonl_path}")
            with open(output_jsonl_path, 'a', encoding='utf-8') as writer:
                with open(input_jsonl_path, 'r', encoding='utf-8') as reader:
                    i = 0
                    for line in reader:
                        if line.strip():
                            task_data = json.loads(line)
                            if i < SKIP_COUNT:
                                i += 1
                                continue
                            self.replay_task(task_data, writer)
                            i += 1
        except FileNotFoundError:
            logging.critical(f"輸入檔案未找到: {input_jsonl_path}")
        except json.JSONDecodeError as e:
            logging.critical(f"解碼 JSON 時出錯: {e}")
        
        logging.info("檔案中的所有任務都已處理完畢。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在星露谷物語中回放動作序列，並將結果輸出到單一的 JSONL 檔案中。")
    parser.add_argument("--input_file", type=str, default="/Users/xspadex/iclr/traj_data/extracted/gpt41_1.jsonl")
    parser.add_argument("--port", type=int, default=10783, help="遊戲伺服器的端口號。")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="保存輸出數據（jsonl 和 images）的根目錄。")
    
    args = parser.parse_args()


    input_paths = [
        "/Users/xspadex/iclr/traj_data/extracted_actions/all_short.jsonl",
    ]
    # input_paths = [
    #     "/Users/xspadex/iclr/traj_data/full_success_traj_actions/craft_only.jsonl",
    # ]
    for input_file in input_paths:
        replayer = ActionReplayer(
            port=args.port, 
            output_dir=args.output_dir
        )
        
        try:
            replayer.run_from_file(input_file)
        except KeyboardInterrupt:
            logging.info("接收到用戶中斷信號，正在退出...")
        finally:
            logging.info("程序已退出。")