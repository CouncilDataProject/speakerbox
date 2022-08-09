#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for tests! There are a whole list of hooks you can define in this file to
run before, after, or to mutate how tests run. Commonly for most of our work, we use
this file to define top level fixtures that may be needed for tests throughout multiple
test files.

In this case, while we aren't using this fixture in our tests, the prime use case for
something like this would be when we want to preload a file to be used in multiple
tests. File reading can take time, so instead of re-reading the file for each test,
read the file once then use the loaded content.

Docs: https://docs.pytest.org/en/latest/example/simple.html
      https://docs.pytest.org/en/latest/plugins.html#requiring-loading-plugins-in-a-test-module-or-conftest-file
"""

from pathlib import Path

import pytest
from pytest import Parser

from speakerbox.utils import _unpack_zip

###############################################################################


@pytest.fixture(scope="session")
def data_dir() -> Path:
    d_dir = Path(__file__).parent / "data"

    # Check for diarized audio
    diarized_dir = d_dir / "diarized"
    if not diarized_dir.exists():
        _unpack_zip(d_dir / "diarized.zip", diarized_dir)

    return d_dir


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--cpu", action="store_true", dest="use_cpu")
    parser.addoption("--ci", action="store_true", dest="ci")
