#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports should be grouped into:
# Standard library imports
# Related third party imports
# Local application / relative imports
# in that order

# Standard library
import logging
from typing import Any, Tuple

# Third party

# Relative

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Example(object):
    """
    This is an example object. Use this example for ideas on how to write doc strings,
    use logging, construct objects. This is not an exhaustive example but a decent
    start.

    Parameters
    ----------
    init_value: int
        An integer value to initialize the object with.
    """

    # Static methods are available to the user regardless of if they have initialized
    # an instance of the class. They are useful when you have small portions of code
    # that while relevant to the class may not depend on entire class state.
    # In this case, this function isn't incredibly valuable outside of the usage of
    # this class and therefore we use the "Python" standard of prefixing the method
    # with an underscore.
    @staticmethod
    def _check_value(val: Any):
        """
        Check that the value is an integer. If not, raises a ValueError.
        """
        if not isinstance(val, int):
            raise ValueError(
                f"Provided value: {val} (type: {type(val)}, is not an integer.)"
            )

    def __init__(self, init_value: int = 10):
        # Check initial value
        self._check_value(init_value)

        # Set values
        self.current_value = init_value
        self.old_value = None

    def update_value(self, new_value: int) -> int:
        """
        Save old value and set new value.

        Parameters
        ----------
        new_value: int
            The new value to assign to the object.

        Returns
        -------
        old_value: int
            The previous value stored in the object.
        """
        # Check new value before assign
        self._check_value(new_value)

        # Passed, now assign
        self.old_value = self.current_value
        self.current_value = new_value
        log.info(f"Updating value from {self.old_value} to {self.current_value}")
        return self.old_value

    def get_value(self) -> int:
        """
        Get the current value.

        Returns
        -------
        current_value: int
            The current value stored in the object.
        """
        return self.current_value

    def get_previous_value(self) -> int:
        """
        Get the previous value.

        Returns
        -------
        previous_value: int
            The previous value stored in the object.
        """
        return self.old_value

    # And example of a property accessor
    # When using this object, this "function" can be called using attribute style
    # These are useful when you want to hide that computation or IO is happening from
    # the user. Usually, these are used when you need to lazy load something or have
    # "immutability" of an object property.
    # ```
    # e = Example(10)
    # stored = e.values
    # ```
    @property
    def values(self) -> Tuple[int]:
        """
        Get both values stored in the object as a tuple of integers.

        Returns
        -------
        values: Tuple[int]
            The current and old values stored in a tuple in (current_value, old_value)
            order.
        """
        return (self.current_value, self.old_value)

    def __str__(self):
        return f"<Example [current: {self.current_value}, previous: {self.old_value}]>"

    # Representation's (reprs) are useful when using interactive Python sessions or
    # when you log an object. They are the shorthand of the object state. In this case,
    # our string method provides a good representation.
    def __repr__(self):
        return str(self)
