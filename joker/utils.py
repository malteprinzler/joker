import argparse
from accelerate.logging import get_logger
import os
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

logger = get_logger(__name__)

import importlib


def import_from(module, obj_name):
    """
    mimics behavior of 'from <module> import <obj>'
    :param module:
    :param obj:
    :return:
    """
    pkg = importlib.import_module(module)
    obj = pkg.__dict__[obj_name]
    return obj


def import_obj(s: str):
    """
    directly returns object from module. E.g 'path.to.package.object' returns 'object'
    :param s:
    :return:
    """
    module, obj_name = ".".join(s.split(".")[:-1]), s.split(".")[-1]
    obj = import_from(module, obj_name)
    return obj


def str2bool(s):
    s = s.lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise ValueError


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
