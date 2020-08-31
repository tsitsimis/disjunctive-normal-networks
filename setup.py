from setuptools import setup


def readme():
    with open("README.md", encoding="UTF-8") as readme_file:
        return readme_file.read()


setup(
    name="disjunctive-nn",
    version="0.1.0",
    description="PyTorch implementation of Disjunctive Normal Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tsitsimis/neural-disjunctive-normal-networks",
    author="Theodore Tsitsimis",
    author_email="th.tsitsimis@gmail.com",
    license="BSD",
    classifiers=[
        "License :: OSI Approved",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["disjunctive_nn"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.1",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.2",
        "torch>=1.6.0+cpu"
        ]
)