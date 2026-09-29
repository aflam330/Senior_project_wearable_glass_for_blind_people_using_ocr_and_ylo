"""Abstract base class that every detection mode inherits from."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseMode(ABC):

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Analyse a camera frame and return a Bangla/English string to speak,
        or None if nothing worth announcing was found.
        """
        ...

    def activate(self) -> None:
        """Called when the user switches INTO this mode."""

    def deactivate(self) -> None:
        """Called when the user switches AWAY from this mode."""

    def cleanup(self) -> None:
        """Called once on application shutdown."""
