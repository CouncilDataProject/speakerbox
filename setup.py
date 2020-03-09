#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "codecov",
    "flake8",
    "black",
    "pytest",
    "pytest-cov",
    "pytest-raises",
]

setup_requirements = [
    "pytest-runner",
]

dev_requirements = [
    "bumpversion>=0.5.3",
    "coverage>=5.0a4",
    "flake8>=3.7.7",
    "ipython>=7.5.0",
    "m2r>=0.2.1",
    "pytest>=4.3.0",
    "pytest-cov==2.6.1",
    "pytest-raises>=0.10",
    "pytest-runner>=4.4",
    "Sphinx>=2.0.0b1",
    "sphinx_rtd_theme>=0.1.2",
    "tox>=3.5.2",
    "twine>=1.13.0",
    "wheel>=0.33.1",
]

interactive_requirements = [
    "altair",
    "jupyterlab",
    "matplotlib",
]

requirements = [
    # stepworkflow requires
    "bokeh",
    "dask[bag]",
    "datastep>=0.1.5",
    "distributed",
    "docutils<0.16",  # needed for botocore (quilt dependency)
    "fire",
    "prefect",
    "python-dateutil<=2.8.0",  # need <=2.8.0 for quilt3 in step
    # project requires
    "pandas",
]

data_requirements = [
    "cdptools[all]>=2.0.5",
    "pydub>=0.23.1",
]

extra_requirements = {
    "test": test_requirements,
    "setup": setup_requirements,
    "dev": dev_requirements,
    "interactive": interactive_requirements,
    "data": data_requirements,
    "all": [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements,
        *data_requirements,
    ]
}

setup(
    author="Jackson Maxfield Brown",
    author_email="jmaxfieldbrown@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description=(
        "A workflow for preparing and training from an audio + transcript "
        "dataset for speaker classification"
    ),
    entry_points={
        "console_scripts": [
            "speakerbox=speakerbox.bin.cli:cli",
        ]
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="speakerbox",
    name="speakerbox",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite="speakerbox/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/CouncilDataProject/speakerbox",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.1.0",
    zip_safe=False,
)
