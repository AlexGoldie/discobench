import pathlib


def get_domains() -> list[str]:
    """Function to see all domains available in DiscoBench registration.

    Returns:
        List of [domains].
    """
    task_path = pathlib.Path("discobench/tasks")
    domains = [p.name for p in task_path.iterdir()]
    return domains
