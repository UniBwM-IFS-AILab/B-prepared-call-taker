"""Runtime plugin discovery for policies and frontends."""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from inspect import isclass

from ems_prepared.model.contracts import (
    FrontendPlugin,
    PolicyFactory,
)


def _iter_group(group: str) -> list[EntryPoint]:
    """Return one entry-point group while validating duplicate names."""
    loaded: list[EntryPoint] = []
    seen_names: set[str] = set()
    for entry_point in entry_points(group=group):
        if entry_point.name in seen_names:
            raise ValueError(
                f"Duplicate entry point name '{entry_point.name}' in group '{group}'."
            )
        seen_names.add(entry_point.name)
        loaded.append(entry_point)
    return loaded


def load_policy_factories(
    group: str = "ems_prepared.policies",
) -> dict[str, PolicyFactory]:
    """Load policy factory plugins from the configured entry-point group."""
    factories: dict[str, PolicyFactory] = {}
    for entry_point in _iter_group(group):
        loaded_obj = entry_point.load()
        if not isinstance(loaded_obj, PolicyFactory):
            if not callable(loaded_obj):
                raise TypeError(
                    f"Entry point '{entry_point.name}' in group '{group}' must resolve to a callable."
                )
            raise TypeError(
                f"Entry point '{entry_point.name}' in group '{group}' must satisfy protocol "
                "'PolicyFactory'."
            )
        factories[entry_point.name] = loaded_obj
    return factories


def load_frontend_plugins(
    group: str = "ems_prepared.frontends",
) -> dict[str, FrontendPlugin]:
    """Load frontend plugins from the configured entry-point group."""
    loaded_plugins: dict[str, FrontendPlugin] = {}
    for entry_point in _iter_group(group):
        loaded_obj = entry_point.load()
        if isclass(loaded_obj):
            try:
                loaded_obj = loaded_obj()
            except TypeError as exc:
                raise TypeError(
                    f"Entry point '{entry_point.name}' in group '{group}' must be "
                    "instantiable without arguments."
                ) from exc
        if not isinstance(loaded_obj, FrontendPlugin):
            raise TypeError(
                f"Entry point '{entry_point.name}' in group '{group}' must satisfy protocol "
                "'FrontendPlugin'."
            )
        loaded_plugins[entry_point.name] = loaded_obj
    return loaded_plugins
