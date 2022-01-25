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

import json
from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


# Fixtures can simply be added as a parameter to the other test or fixture functions to
# expose them. If we had multiple tests that wanted to use the contents of this file,
# we could simply add "loaded_example_values" as a parameter for each test.
@pytest.fixture
def loaded_example_values(data_dir) -> Dict[str, int]:
    with open(data_dir / "example_values.json", "r") as read_in:
        return json.load(read_in)
