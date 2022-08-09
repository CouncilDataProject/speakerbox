#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

seattle_reqs = [
    "cdp-backend~=3.0.3",
]

test_requirements = [
    "black>=22.3.0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "isort>=5.7.0",
    "mypy>=0.790",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "tox>=3.15.2",
    *seattle_reqs,
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "jupyterlab>=3.2.8",
    "m2r2>=0.2.7",
    "Sphinx>=3.4.3",
    "furo>=2022.4.7",
    "twine>=3.1.1",
    "wheel>=0.34.2",
    # Pins until fixed
    "docutils>=0.18,<0.19",
]

requirements = [
    "dataclasses_json~=0.5",
    "datasets[audio]~=1.18",
    "librosa~=0.8",
    "matplotlib~=3.5",
    "pandas~=1.0",
    "pydub~=0.25",
    "scikit-learn~=1.0",
    "speechbrain~=0.5.11",
    "torch~=1.10",
    "torchaudio~=0.10",
    "transformers~=4.16",
    "pyannote.audio~=2.0",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "seattle": seattle_reqs,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="Eva Maxfield Brown",
    author_email="evamaxfieldbrown@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Speaker Annotation for Transcripts using Audio Classification",
    entry_points={
        "console_scripts": [],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="speakerbox",
    name="speakerbox",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
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
