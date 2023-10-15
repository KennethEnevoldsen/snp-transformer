from dataclasses import fields
from typing import Any, OrderedDict

from torch import Tensor


class DataclassAsDict(OrderedDict[str, Tensor]):
    """
    Dataclasses that derive from this struct are also a dictionary.

    Since this class should only be used for dataclasses that are
    ``torch.jit.trace``-able, only ``Tensor`` fields are supported.

    Only dataclass fields and keys corresponding to those fields
    can be changed. Fields and keys cannot be removed.

    Derived from https://github.com/explosion/curated-transformers/blob/main/curated_transformers/util/dataclass.py#L7
    """

    def __post_init__(self):
        for field in fields(self):  # type: ignore
            value = getattr(self, field.name)
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"`DataclassAsDict` only supports `Tensor` members, but field '{field.name}' has type `{field.type.__name__}`"
                )

            super().__setitem__(field.name, value)

    def __delitem__(self, key: str):
        raise NotImplementedError()

    def __delattr__(self, name: str):
        raise NotImplementedError()

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field '{name}' cannot be set to non-Tensor type `{type(value).__name__}`"
            )

        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Tensor) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Key cannot be set to non-`str` type `{type(key).__name__}`"
            )

        if not isinstance(value, Tensor):
            raise TypeError(
                f"Field '{key}' cannot be set to non-Tensor type `{type(value).__name__}`"
            )

        super().__setattr__(key, value)
        super().__setitem__(key, value)
