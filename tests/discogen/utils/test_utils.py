import os

UTILS_PATH = "discogen/utils"


def test_exist() -> None:
    """Ensure description is in discogen/utils."""
    expected_files = [
        "__init__.py",
        "description.md",
        "get_domains.py",
        "get_modules.py",
        "make_files.py",
        "run_main.py",
    ]

    for file in expected_files:
        assert os.path.exists(f"{UTILS_PATH}/{file}")
