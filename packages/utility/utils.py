from typing import Callable, Any, List

from functools import reduce


def apply_pipeline(data: Any, functions: List[Callable]):
    if len(functions) == 0:
        return data

    return reduce(lambda prev, func: func(prev), functions, data)
