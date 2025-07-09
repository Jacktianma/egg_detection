# config/config.py
import yaml
import os

def load_config(config_path="config/config.yaml"):
    """从 YAML 文件加载配置"""
    try:
        # 确保路径相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, config_path)
        with open(full_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {full_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"解析 YAML 文件出错: {e}")