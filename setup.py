from setuptools import setup, find_packages

setup(
    name="deephedging",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "tensorflow-probability",
        "numpy",
        "pandas",
        "matplotlib",
        "cdxbasics",
    ],
)