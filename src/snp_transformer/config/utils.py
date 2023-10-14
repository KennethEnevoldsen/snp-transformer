from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Optional, Union

from confection import Config

from snp_transformer import Registry
from snp_transformer.config.config_schemas import ResolvedConfigSchema

default_config_path = Path(__file__).parent / "default_config.cfg"


def load_config(config_path: Optional[Path] = None) -> Config:
    if config_path is None:
        config_path = default_config_path
    cfg = Config().from_disk(config_path)
    return cfg


def parse_config(config: Config) -> ResolvedConfigSchema:
    resolved = Registry.resolve(config)
    return ResolvedConfigSchema(**resolved)


def flatten_nested_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = ".",
) -> dict:
    """Recursively flatten an infinitely nested config. E.g. {"level1":

    {"level2": "level3": {"level4": 5}}}} becomes:

    {"level1.level2.level3.level4": 5}.

    Args:
        d (Union[Dict, MutableMapping]): Dict to flatten.
        parent_key (str): The parent key for the current dict, e.g. "level1" for the
            first iteration. Defaults to "".
        sep (str): How to separate each level in the dict. Defaults to ".".

    Returns:
        Dict: The flattened dict.
    """

    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(d=v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
