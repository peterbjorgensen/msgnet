from setuptools import find_packages, setup, Extension

setup(
    name="msgnet",
    version="0.0.1",
    description="Python library for implementation of message passing neural networks with example scripts",
    author="Peter Bjørn Jørgensen",
    author_email="peterbjorgensen@gmail.com",
    # url = 'https://docs.python.org/extending/building',
    package_dir={"": "src"},
    packages=find_packages("src"),
)
