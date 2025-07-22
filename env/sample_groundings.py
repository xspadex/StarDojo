import os
import json
import uuid
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

# 假设你的环境代码位于一个名为 stardew_env 的包中
# 如果不是，请相应地修改导入路径
from stardew_env import StarDojo

# --- 配置区 ---

# 1. 定义你想要去采集数据的地图列表
#    确保这些名称与游戏中的 Location Name 完全一致
MAPS_TO_VISIT = [
    "Farm",
    "Town",
    "Mountain",
    "Forest",
    "Beach",
    "Desert"
]

# 2. 定义输出数据的文件夹
OUTPUT_DIR = "./vlm_grounding_data"

# 3. 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DataCollector:
    """
    用于从星露谷物语环境中自动采集VLM grounding数据的类。
    """
    def __init__(self, port: int, save_index: int, image_save_path: str, repeat_num: int):
        """
        初始化数据采集器。

        Args:
            port (int): 游戏服务器端口。
            save_index (int): 游戏存档索引。
            image_save_path (str): 截图保存的根路径。
            repeat_num (int): 在每个地图上随机传送和采集的次数。
        """
        self.output_path = OUTPUT_DIR
        self.screenshots_path = os.path.join(self.output_path, "screenshots")
        self.metadata_path = os.path.join(self.output_path, "metadata.jsonl")
        self.repeat_num = repeat_num
        # 创建输出目录
        os.makedirs(self.screenshots_path, exist_ok=True)
        
        logging.info("正在连接到星露谷物语环境...")
        self.env = StarDojo(
            port=port,
            save_index=save_index,
            new_game=False,
            image_save_path=self.screenshots_path,
            output_video=False,
            max_image_storage=1
        )
        
        self.action_proxy = self.env.action_proxy
        self.metadata = []
        
        self.action_proxy.wait_for_server()
        logging.info("环境连接成功！")

    def get_last_part(self, s: str) -> str:
        """从 'Type.Subtype.Name' 格式的字符串中提取最后一部分。"""
        if isinstance(s, str) and '.' in s:
            return s.split('.')[-1]
        return s

    def extract_objects_from_tile(self, tile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从单个地块数据中提取所有感兴趣的物品信息。
        
        Args:
            tile (Dict): 来自观测数据的单个地块信息。

        Returns:
            List[Dict]: 包含物品信息的字典列表，每个字典含 'name' 和 'position'。
        """
        objects = []
        position = tile.get('position')
        if not position:
            return []

        # --- MODIFIED: 添加了 'building_info' 和 'exit_info' ---
        # 在这里添加或删除你感兴趣的物品类型
        object_keys = {
            'debris_at_tile',    # 杂物 (石头, 木头, 杂草)
            'object_at_tile',    # 放置的物品或可采集的物品
            'crop_at_tile',      # 农作物
            'terrain_at_tile',   # 例如树木
            'door_info'          # 地图出口或建筑的门 (例如 "Town", "FarmHouse")
        }

        for key in object_keys:
            obj_data = tile.get(key)
            if obj_data:
                obj_name = ""
                # 根据不同key的数据结构提取名称
                if key == 'crop_at_tile' and isinstance(obj_data, dict):
                    # 假设作物信息是一个字典
                    obj_name = obj_data.get('seed_name', 'Unknown Crop')
                elif isinstance(obj_data, str):
                    obj_name = self.get_last_part(obj_data)
                elif key == 'exit_info' and isinstance(obj_data, str):
                    obj_name = 'exit to ' + obj_data
                
                if obj_name:
                    objects.append({'name': obj_name, 'position': position})
        
        return objects

    def scan_current_map_and_collect_data(self):
        """
        扫描当前屏幕可见区域，为每个发现的物品生成并保存一条数据。
        """
        current_location = self.env._get_obs().get('location', 'Unknown Location')
        logging.info(f"正在扫描地图: {current_location}")
        
        try:
            obs = self.env._get_obs()
        except Exception as e:
            logging.error(f"获取观测数据时出错: {e}")
            return

        # --- MODIFIED: 优先使用 'viewingtiles' ---
        tiles_to_scan = obs.get('viewingtiles')
        if tiles_to_scan is None:
            logging.warning("'viewingtiles' not found in observation, falling back to 'surroundingsdata'.")
            tiles_to_scan = obs.get('surroundingsdata', [])

        player_pos = obs.get('player', {}).get('position')
        
        if not tiles_to_scan or not player_pos:
            logging.warning("观测数据不完整(缺少地块或玩家位置)，跳过此次扫描。")
            return

        # --- OPTIMIZED: 每轮扫描只保存一张截图 ---
        rgb_image = obs['screenshot'][:, :, :3].astype(np.uint8)
        image_filename = f"{uuid.uuid4().hex}.jpeg"
        full_image_path = os.path.join(self.screenshots_path, image_filename)
        
        try:
            img = Image.fromarray(rgb_image)
            img.save(full_image_path, 'JPEG')
        except Exception as e:
            logging.error(f"保存截图失败: {full_image_path}. 错误: {e}")
            return
        
        collected_count = 0
        for tile in tiles_to_scan:
            objects_on_tile = self.extract_objects_from_tile(tile)
            
            for obj in objects_on_tile:
                # 所有物品都关联到本次扫描生成的同一张截图
                data_record = {
                    "image_file": os.path.join("screenshots", image_filename), # 使用相对路径
                    "object_name": obj['name'],
                    "object_position": obj['position'],
                    "player_position": player_pos,
                    "map_name": current_location
                }
                
                # 直接将JSON对象写入文件，避免内存占用过大
                with open(self.metadata_path, 'a') as f:
                    f.write(json.dumps(data_record) + '\n')
                
                collected_count += 1
        
        logging.info(f"扫描完成，在本区域新收集了 {collected_count} 条数据。")


    def run_collection(self):
        """
        执行完整的数据采集流程。
        """
        for map_name in MAPS_TO_VISIT:
            logging.info(f"--- 前往地图: {map_name} ---")
            try:
                self.action_proxy.warp(map_name, 10, 10) # 传送到地图的(10,10)初始位置
                time.sleep(3) # 等待场景加载
            except AttributeError:
                logging.error("致命错误: 'warp' 技能不存在。请检查你的环境API。")
                break
            except Exception as e:
                logging.error(f"传送到 {map_name} 时发生错误: {e}")
                continue

            for i in range(self.repeat_num):
                logging.info(f"在 {map_name} 进行第 {i+1}/{self.repeat_num} 次随机采集...")
                # 随机传送以获得不同的视角
                self.action_proxy.teleport_random() 
                # 等待一小段时间，确保屏幕内容已更新
                time.sleep(0.5) 
                self.scan_current_map_and_collect_data()
        
        self.cleanup()


    def cleanup(self):
        """
        清理资源并退出。
        """
        logging.info("数据采集完成。正在关闭与环境的连接...")
        self.env.exit()
        logging.info("程序已退出。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为VLM训练批量生成星露谷物语的Grounding数据")
    parser.add_argument("--port", type=int, default=10783, help="游戏服务器的端口号")
    parser.add_argument("--save_index", type=int, default=0, help="游戏存档的索引")
    parser.add_argument("--repeat_num", type=int, default=200, help="在每个地图上随机传送和采集的次数")
    
    args = parser.parse_args()

    collector = DataCollector(
        port=args.port, 
        save_index=args.save_index,
        image_save_path=os.path.join(OUTPUT_DIR, "screenshots"), # 内部管理路径
        repeat_num=args.repeat_num
    )
    
    try:
        collector.run_collection()
    except KeyboardInterrupt:
        logging.info("接收到用户中断信号。")
    finally:
        collector.cleanup()