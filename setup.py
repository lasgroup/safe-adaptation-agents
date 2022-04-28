#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'safe_adaptation_gym', 'jax',
    'pip install git+https://github.com/deepmind/dm-haiku',
    'optax', 'jmp', 'numpy', 'ruamel.yaml', 'tensorboardX'
]

setup(
    name='safe_adaptation_agents',
    version='0.0.0',
    packages=find_packages(),
    python_requires='>3.8',
    include_package_data=True,
    install_requires=required)
