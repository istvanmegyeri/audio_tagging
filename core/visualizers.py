from abc import ABC, abstractmethod
from argparse import ArgumentParser


class DEFAULT(ABC):
    def __init__(self, args) -> None:
        self.params = self.get_parser().parse_args(args)

    @abstractmethod
    def get_parser(self) -> ArgumentParser:
        pass
