"""Command line interface for DiscoBench."""

import os

import click
import yaml

from discobench import create_config, create_task, get_domains, get_modules


@click.group()
def cli() -> None:
    """DiscoBench - Modular Benchmark for Automated Algorithm Discovery."""
    pass


@cli.command("create-task")
@click.option("--task-domain", type=str, required=True, help="The task domain to create the task for.")
@click.option("--test", is_flag=True, help="If passed, create test task instead of training task.")
@click.option("--example", is_flag=True, default=None, help="If passed, use example task config rather than your own.")
@click.option(
    "--config-path",
    type=str,
    help="The path to your task_config.yaml. If not provided, this will default to the DiscoBench task_config.yaml for your provided task_domain.",
)
@click.option(
    "--no-data",
    is_flag=True,
    help="If passed, will create the task without downloading the data. The task code will generally not be able to run, but this will allow you to see how the code looks for a specific task.",
)
def create_task_cmd(
    task_domain: str,
    test: bool,
    config_path: str | None = None,
    example: bool | None = None,
    no_data: bool | None = None,
) -> None:
    """Create task source files for a specified task domain."""
    create_task(task_domain=task_domain, test=test, config_path=config_path, example=example, no_data=no_data)
    mode = "test" if test else "training"
    click.echo(f"Successfully created {mode} task for domain: {task_domain}.")


@cli.command("get-domains")
def get_domains_cmd() -> None:
    """List all available task domains in DiscoBench."""
    domains = get_domains()
    click.echo("\n".join(domains))


@cli.command("get-modules")
def get_modules_cmd() -> None:
    """List all available modules for a specified task domain."""
    module_dict = get_modules()
    for domain, modules in module_dict.items():
        click.echo(f"{domain}: {', '.join(modules)}")


@cli.command("create-config")
@click.option("--task-domain", type=str, required=True, help="The task domain to create the task for.")
@click.option(
    "--save-dir", type=str, required=False, default="task_configs", help="The directory to save the config to."
)
def create_config_cmd(task_domain: str, save_dir: str) -> None:
    """Save default config for editing for a specified task domain."""
    config = create_config(task_domain)

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/task_config_{task_domain}.yml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":
    cli()
