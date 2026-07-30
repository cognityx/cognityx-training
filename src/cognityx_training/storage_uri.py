"""Authoritative resolution of provider-neutral Storage URIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class StorageUriResolution:
    """One URI bound to its physical profile and logical namespace."""

    uri: str
    store: Any
    key: str
    detected_namespace: str
    selected_role: str
    selected_profile: str
    shared_compatibility: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "detected_namespace": self.detected_namespace,
            "selected_role": self.selected_role,
            "selected_profile": self.selected_profile,
            "shared_compatibility": self.shared_compatibility,
        }


class StorageUriResolutionError(ValueError):
    """Structured failure to bind a Storage URI safely."""

    def __init__(
        self,
        uri: str,
        *,
        detected_namespace: str | None,
        selected_role: str | None,
        selected_profile: str | None,
        failure_category: str,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.uri = uri
        self.detected_namespace = detected_namespace
        self.selected_role = selected_role
        self.selected_profile = selected_profile
        self.failure_category = failure_category
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "detected_namespace": self.detected_namespace,
            "selected_role": self.selected_role,
            "selected_profile": self.selected_profile,
            "failure_category": self.failure_category,
            "detail": self.detail,
        }


def resolve_storage_uri(
    runtime: Any,
    uri: str,
    *,
    role_override: str | None = None,
) -> StorageUriResolution:
    """Resolve a Storage URI without reinterpreting its namespace."""
    parsed = urlparse(uri)
    if parsed.scheme != "storage" or not parsed.netloc:
        raise StorageUriResolutionError(
            uri,
            detected_namespace=None,
            selected_role=role_override,
            selected_profile=None,
            failure_category="invalid_storage_uri",
            detail=f"Expected provider-neutral storage:// URI: {uri}",
        )
    path = parsed.path.lstrip("/")
    namespace, _, remainder = path.partition("/")
    if not namespace:
        raise StorageUriResolutionError(
            uri,
            detected_namespace=None,
            selected_role=role_override,
            selected_profile=parsed.netloc,
            failure_category="missing_namespace",
            detail=f"Storage URI has no logical namespace: {uri}",
        )
    if parsed.netloc == "shared":
        return _resolve_shared(runtime, uri, path)

    namespace_roles = {
        role.namespace.strip("/"): role_name
        for role_name, role in runtime.config.roles.items()
        if role.namespace.strip("/")
    }
    selected_role = namespace_roles.get(namespace)
    if selected_role is None:
        if role_override is None:
            raise StorageUriResolutionError(
                uri,
                detected_namespace=namespace,
                selected_role=None,
                selected_profile=parsed.netloc,
                failure_category="unknown_namespace",
                detail=(
                    f"Storage URI namespace '{namespace}' is not configured: {uri}"
                ),
            )
        selected_role = role_override
        key = path
    else:
        if role_override is not None and role_override != selected_role:
            raise StorageUriResolutionError(
                uri,
                detected_namespace=namespace,
                selected_role=role_override,
                selected_profile=parsed.netloc,
                failure_category="role_namespace_mismatch",
                detail=(
                    f"Storage URI namespace '{namespace}' maps to role "
                    f"'{selected_role}', not override '{role_override}'"
                ),
            )
        key = remainder
    try:
        store = runtime.for_profile(parsed.netloc, role_name=selected_role)
    except Exception as exc:
        raise StorageUriResolutionError(
            uri,
            detected_namespace=namespace,
            selected_role=selected_role,
            selected_profile=parsed.netloc,
            failure_category="profile_or_role_unavailable",
            detail=str(exc),
        ) from exc
    if not key:
        raise StorageUriResolutionError(
            uri,
            detected_namespace=namespace,
            selected_role=selected_role,
            selected_profile=parsed.netloc,
            failure_category="missing_storage_key",
            detail=f"Storage URI does not identify an object: {uri}",
        )
    return StorageUriResolution(
        uri=uri,
        store=store,
        key=key,
        detected_namespace=namespace,
        selected_role=selected_role,
        selected_profile=parsed.netloc,
    )


def _resolve_shared(runtime: Any, uri: str, key: str) -> StorageUriResolution:
    profile = runtime.config.default_profile
    if not profile:
        raise StorageUriResolutionError(
            uri,
            detected_namespace="shared",
            selected_role="shared",
            selected_profile=None,
            failure_category="default_profile_unavailable",
            detail="Legacy shared URI requires a configured default Storage profile",
        )
    role_name = next(
        (
            name
            for name, role in runtime.config.roles.items()
            if role.profile == profile
        ),
        None,
    )
    if role_name is None:
        raise StorageUriResolutionError(
            uri,
            detected_namespace="shared",
            selected_role="shared",
            selected_profile=profile,
            failure_category="default_profile_unavailable",
            detail=f"No role is bound to default Storage profile '{profile}'",
        )
    try:
        bound = runtime.for_profile(profile, role_name=role_name)
        shared = bound._client.for_shared_data()
    except Exception as exc:
        raise StorageUriResolutionError(
            uri,
            detected_namespace="shared",
            selected_role="shared",
            selected_profile=profile,
            failure_category="shared_scope_unavailable",
            detail=str(exc),
        ) from exc
    return StorageUriResolution(
        uri=uri,
        store=shared,
        key=key,
        detected_namespace="shared",
        selected_role="shared",
        selected_profile=profile,
        shared_compatibility=True,
    )
