#!/usr/bin/env python
import os

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="UTF-8") as f:
    readme = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines()]

package_data = {}
for root, dirs, files in os.walk(os.path.join("nebula", "objects")):
    for file in files:
        package_data.setdefault('objects', []).append(os.path.join('objects', file))

version = "0.0.6"
setup(
    name="nebula",
    author="Dmitrijs Trizna",
    author_email="d.trizna@pm.me",
    version=version,
    description="Dynamic Malware Analysis Model",
    long_description=readme,
    homepage="https://github.com/dtrizna/nebula",
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)