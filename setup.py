"""simple-captcha."""

from pathlib import Path

from setuptools import find_packages, setup


def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()


BASE_DIR = Path(__file__).parent


setup(
    name="simple-captcha",
    long_description=open(BASE_DIR / "README.md").read(),
    install_requires=read_lines(BASE_DIR / "requirements.txt"),
    extras_require={"dev": read_lines(BASE_DIR / "requirements_dev.txt")},
    packages=find_packages(exclude=["docs"]),
    version="0.1.0",
    description="simple-captcha: Generating Diverse and Realistic Driving Scenarios from Scratch",
    author="Shuo",
    license="proprietary",
)
