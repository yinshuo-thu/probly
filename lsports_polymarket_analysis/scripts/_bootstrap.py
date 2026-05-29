"""把项目根目录加入 sys.path, 使脚本可从任意位置 `python scripts/xxx.py` 运行。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
