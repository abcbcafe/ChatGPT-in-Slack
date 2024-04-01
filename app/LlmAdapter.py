import abc
from abc import ABC


class LlmAdapter(ABC):
    @abc.abstractmethod
    def generate_text(self, messages):
        pass


