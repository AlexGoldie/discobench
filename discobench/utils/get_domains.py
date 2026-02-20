import pathlib


def get_domains() -> list[str]:
    """Function to see all domains available in DiscoBench registration.

    Returns:
        List of [domains].
    """
    task_path = pathlib.Path(__file__).parent.parent / "tasks"
    # TODO: Should filter with p.is_dir() to exclude __pycache__, .DS_Store, etc. Not sure if this happens in practice though?
    domains = [p.name for p in task_path.iterdir()]
    return domains
