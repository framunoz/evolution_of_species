import pathlib

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

# List of requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

setup(
    name="perthame_pde",
    packages=find_packages(),
    version="0.1.0",
    description="Library to solve the Perthame's equations",
    author="J. Bravo, V. Gómez & F. Muñoz",
    # license=None,
    install_requires=install_requires,
)
