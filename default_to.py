from typing import Callable, Optional, TypeVar

T = TypeVar('T')


def default_to_fn(optional_value: Optional[T], make_default: Callable[[], T]) -> T:
    return make_default() if optional_value is None else optional_value


def default_to(optional_value: Optional[T], default: T) -> T:
    return default_to_fn(optional_value, lambda: default)
