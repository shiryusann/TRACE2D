"""
Base feature definition
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mmdemo2d.base_interface import BaseInterface

T = TypeVar("T", bound=BaseInterface)


class BaseFeature(ABC, Generic[T]):
    """
    The base class all features in the demo must implement.
    """

    def __init__(self, *args) -> None:
        self._deps = []
        self._rev_deps = []
        self._register_dependencies(args)

    def _register_dependencies(self, deps: "list[BaseFeature] | tuple"):
        """
        Add other features as dependencies which are required
        to be evaluated before this feature.

        Arguments:
        deps -- a list of dependency features
        """
        assert len(self._deps) == 0, "Dependencies have already been registered"
        for d in deps:
            self._deps.append(d)
            d._rev_deps.append(self)

    @abstractmethod
    def get_output(self, *args, **kwargs) -> T | None:
        """
        Return output of the feature. The return type must be the output
        interface to provide new data and `None` if there is no new data.
        It is very important that this function does not modify any of the
        input interfaces because they may be reused for other features.

        Arguments:
        args -- list of output interfaces from dependencies in the order
                they were registered. Calling `.is_new()` on any of these
                elements will return True if the argument has not been
                sent before. It is possible that the interface will not
                contain any data before the first new data is sent.
        """
        raise NotImplementedError

    def initialize(self):
        """
        Initialize feature. This is where all the time/memory
        heavy initialization should go. Put it here instead of
        __init__ to avoid wasting resources when extra features
        exist. This method is guaranteed to be called before
        the first `get_output` and run on the main thread.
        """

    def finalize(self):
        """
        Perform any necessary cleanup. This method is guaranteed
        to be called after the final `get_output` and run on the
        main thread.
        """

    def is_done(self) -> bool:
        """
        Return True if the demo should exit. This will
        always return False if not overridden.
        """
        return False
