from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.asr_lib.models.base_asr import BaseASR

T = TypeVar("T")


def requires_model(method: Callable[..., T]) -> Callable[..., T]:
    """Ensure that a method is only called when the model is initialized.

    Parameters
    ----------
    method : Callable[..., T]
        The method to decorate.

    Returns
    -------
    Callable[..., T]
        The decorated method.

    Raises
    ------
    RuntimeError
        If the model is not initialized but the method is called.

    """

    @functools.wraps(method)
    def wrapper(
        self: BaseASR,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> T:
        if not self._model_initialized:
            method_name = method.__name__
            msg = (
                f"Cannot call '{method_name}' because the model was initialized "
                f"in metadata-only mode. Please create a new instance with "
                f"initialize_model=True."
            )
            raise RuntimeError(msg)

        return method(self, *args, **kwargs)

    return wrapper
