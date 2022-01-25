#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

import pytest
from speakerbox import Example


# If you only have a single condition you need to test, a single test is _okay_
# but parametrized tests are encouraged
def test_value_change():
    start_val = 5
    new_val = 20

    example = Example(start_val)
    example.update_value(new_val)
    assert example.get_value() == new_val and example.get_previous_value() == start_val


# Generally, you should parametrize your tests, but you should include exception tests
# like below!
@pytest.mark.parametrize(
    "start_val, next_val, expected_values",
    [
        # (start_val, next_val, expected_values)
        (5, 20, (20, 5)),
        (10, 40, (40, 10)),
        (1, 2, (2, 1)),
    ],
)
def test_parameterized_value_change(start_val, next_val, expected_values):
    example = Example(start_val)
    example.update_value(next_val)
    assert expected_values == example.values


# The best practice would be to parametrize your tests, and include tests for any
# exceptions that would occur
@pytest.mark.parametrize(
    "start_val, next_val, expected_values",
    [
        # (start_val, next_val, expected_values)
        (5, 20, (20, 5)),
        (10, 40, (40, 10)),
        (1, 2, (2, 1)),
        pytest.param(
            "hello",
            None,
            None,
            marks=pytest.mark.raises(
                exception=ValueError
            ),  # Init value isn't an integer
        ),
        pytest.param(
            1,
            "hello",
            None,
            marks=pytest.mark.raises(
                exception=ValueError
            ),  # Update value isn't an integer
        ),
    ],
)
def test_parameterized_value_change_with_exceptions(
    start_val, next_val, expected_values
):
    example = Example(start_val)
    example.update_value(next_val)
    assert expected_values == example.values
