import os
import subprocess
import sys


def run_all_main_py(start_dir: str = ".") -> None:
    """Run all main.py files in the given directory and its subdirectories.

    Args:
        start_dir: The directory to start the search for main.py files from.
    """
    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]

        if "main.py" in files:
            main_path = os.path.join(root, "main.py")
            print(f"Running: {main_path}")
            try:
                subprocess.run([sys.executable, main_path], check=True)  # noqa: S603
            except subprocess.CalledProcessError as e:
                print(f"Error running {main_path}: {e}")


if __name__ == "__main__":
    run_all_main_py()
