# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decorator and table helpers for Python-defined profiles."""

from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from inspect import iscoroutinefunction
from typing import Any, TypeAlias, TypeVar, cast, dataclass_transform

from switchyard.lib.profiles.protocols import Profile, ProfileConfig, ProfileT_co

ConfigClassT = TypeVar("ConfigClassT", bound=type[object])
AnyProfileConfig: TypeAlias = ProfileConfig[Profile[Any]]

_PROFILE_CONFIGS: dict[str, type[AnyProfileConfig]] = {}


class ProfileConfigError(ValueError):
    """Raised when a Python profile config declaration is invalid."""


def _normalize_profile_type(profile_type: str) -> str:
    """Validate and normalize a serialized profile type discriminator."""
    normalized = profile_type.strip()
    if not normalized:
        raise ProfileConfigError("profile type must not be empty")
    return normalized


def _config_class(config_cls: type[object]) -> type[AnyProfileConfig]:
    """Validate a class has the minimum config shape expected by the table."""
    if not is_dataclass(config_cls):
        raise ProfileConfigError(
            f"{config_cls.__qualname__} must be a dataclass or use @profile_config directly"
        )
    if not callable(getattr(config_cls, "build", None)):
        raise ProfileConfigError(f"{config_cls.__qualname__} must define build()")
    return cast(type[AnyProfileConfig], config_cls)


def register_profile_config(
    profile_type: str,
    config_cls: type[AnyProfileConfig],
    *,
    replace: bool = False,
) -> None:
    """Register ``config_cls`` as the config owner for ``profile_type``.

    ``replace`` is deliberately explicit so accidental duplicate profile types
    fail during import, where they are easiest to diagnose.
    """
    normalized = _normalize_profile_type(profile_type)
    validated_cls = _config_class(config_cls)
    existing = _PROFILE_CONFIGS.get(normalized)
    if existing is not None and existing is not validated_cls and not replace:
        raise ProfileConfigError(
            f"profile type {normalized!r} is already registered to {existing.__qualname__}"
        )
    _PROFILE_CONFIGS[normalized] = validated_cls


@dataclass_transform(frozen_default=True)
def profile_config(
    profile_type: str,
    *,
    frozen: bool = True,
    slots: bool = True,
    register: bool = False,
) -> Callable[[ConfigClassT], ConfigClassT]:
    """Decorate and optionally register a Python profile config class.

    The decorator mirrors the Rust ``#[profile_config]`` macro at Python scale:
    it ensures the class is a dataclass, stamps ``PROFILE_TYPE``, requires a
    ``build()`` method, and only mutates the local table when ``register``
    is explicitly enabled. Existing dataclasses are accepted unchanged.
    """
    normalized = _normalize_profile_type(profile_type)

    def decorate(cls: ConfigClassT) -> ConfigClassT:
        config_cls = cls if is_dataclass(cls) else cast(
            ConfigClassT,
            dataclass(frozen=frozen, slots=slots)(cls),
        )
        type.__setattr__(config_cls, "PROFILE_TYPE", normalized)
        typed_config_cls = _config_class(config_cls)
        if register:
            register_profile_config(normalized, typed_config_cls)
        return config_cls

    return decorate


def lookup_profile_config(profile_type: str) -> type[AnyProfileConfig]:
    """Return the registered config class for ``profile_type``."""
    normalized = _normalize_profile_type(profile_type)
    try:
        return _PROFILE_CONFIGS[normalized]
    except KeyError as exc:
        raise ProfileConfigError(f"unknown profile type {normalized!r}") from exc


def registered_profile_configs() -> dict[str, type[AnyProfileConfig]]:
    """Return a shallow copy of registered Python profile config classes."""
    return dict(_PROFILE_CONFIGS)


def registered_profile_config_types() -> tuple[str, ...]:
    """Return registered profile type discriminators in deterministic order."""
    return tuple(sorted(_PROFILE_CONFIGS))


def profile_config_type(config_or_cls: object) -> str:
    """Return the stamped ``PROFILE_TYPE`` for a config instance or class."""
    config_cls = config_or_cls if isinstance(config_or_cls, type) else type(config_or_cls)
    profile_type = getattr(config_cls, "PROFILE_TYPE", None)
    if not isinstance(profile_type, str) or not profile_type:
        raise ProfileConfigError(f"{config_cls.__qualname__} is not a registered profile config")
    return profile_type


def build_profile(config: ProfileConfig[ProfileT_co]) -> ProfileT_co:
    """Build ``config`` and verify the result implements the profile runtime."""
    profile = config.build()
    if not isinstance(profile, Profile):
        raise ProfileConfigError(
            f"{type(config).__qualname__}.build() did not return a Profile runtime with "
            "run/process/rprocess methods"
        )
    for method_name in ("run", "process", "rprocess"):
        method = getattr(profile, method_name, None)
        if not iscoroutinefunction(method):
            raise ProfileConfigError(
                f"{type(config).__qualname__}.build() returned a Profile runtime, but "
                f"{method_name}() must be async"
            )
    return profile


__all__ = [
    "ProfileConfigError",
    "build_profile",
    "lookup_profile_config",
    "profile_config",
    "profile_config_type",
    "register_profile_config",
    "registered_profile_config_types",
    "registered_profile_configs",
]
