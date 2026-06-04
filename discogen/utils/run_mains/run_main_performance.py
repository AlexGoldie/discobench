import json
import os
import subprocess
import sys
from typing import Any


def run_all_main_py(start_dir: str = ".") -> tuple[dict[str, Any], dict[str,str]]:
    """Run all main.py files in the given directory and its subdirectories.

    Args:
        start_dir: The directory to start the search for main.py files from.
        
    Returns:
        results: A dictionary of scores for all successful runs
        errors: A dictionary of error messages for all unsuccessful runs
    """
    results: dict[str, Any] = {}
    errors:  dict[str, str] = {}

    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]

        if "main.py" not in files:
            continue
    
        main_path = os.path.abspath(os.path.join(root, "main.py"))
        print(f"Running: {main_path}")
        try:
            result = subprocess.run([sys.executable, main_path], check=True, capture_output=True, text=True)  # noqa: S603

            metrics = next(
                (json.loads(line) for line in reversed(result.stdout.strip().split("\n"))
                 if line.strip().startswith("{")),
                None,
            )

            if metrics:
                results[root] = metrics
            else:
                errors[root] = result.stdout
        except subprocess.CalledProcessError as e:
            error_message = e.stderr
            errors[root] = error_message
        except json.JSONDecodeError as e:
            errors[root] = f"Failed to parse metrics JSON: {e}"

    print(json.dumps({"results": results, "errors": errors}))
    return results, errors


if __name__ == "__main__":
    run_all_main_py()
