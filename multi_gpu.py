from accelerate.commands.accelerate_cli import *
from omegaconf import OmegaConf
import sys
import argparse

OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_config(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args(args)
    config = OmegaConf.load(args.config)
    return config


def config2argv(config):
    argv = list()
    for k, v in config.items():
        argv += [f"--{k}", f"{v}"]

    return argv


def main(args=None):
    """
    mostly copyied from accelerate/commands/accelerate_cli.py, just added args parameter
    """
    parser = ArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]", allow_abbrev=False)
    subparsers = parser.add_subparsers(help="accelerate command helpers")

    # Register commands
    get_config_parser(subparsers=subparsers)
    env_command_parser(subparsers=subparsers)
    launch_command_parser(subparsers=subparsers)
    tpu_command_parser(subparsers=subparsers)
    test_command_parser(subparsers=subparsers)

    # Let's go
    args = parser.parse_args(args)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    args.func(args)


if __name__ == "__main__":
    """
    runs scripts with multi-gpu support using the accelerate package. Automatically reads accelerate configs from config file.
    execute as: python multi_gpu.py <script_path.py> <config_path.yaml>
    """
    script = sys.argv[1]
    config_path = sys.argv[2]
    config = parse_config([config_path])
    OmegaConf.resolve(config)
    accelerate_config = config["accelerate_kwargs"]
    argv = config2argv(config["accelerate_kwargs"])
    if accelerate_config["num_processes"] > 1:
        argv.append("--multi_gpu")
    argv = ["launch"] + argv + [script, config_path]
    main(argv)
