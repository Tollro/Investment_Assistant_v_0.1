import subprocess
import sys
from pathlib import Path

def main():
    # 确保当前工作目录为项目根目录
    project_root = Path(__file__).parent
    ui_file = project_root / "nodes" / "ui.py"

    if not ui_file.exists():
        print(f"错误：找不到 UI 文件 {ui_file}")
        sys.exit(1)

    # 调用 streamlit run 命令
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_file)]
    print(f"正在启动 Streamlit 应用：{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"运行出错：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()