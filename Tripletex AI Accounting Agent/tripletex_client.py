from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import requests
from openapi_guard import is_restricted_pilot_endpoint, validate_request_contract

logger = logging.getLogger(__name__)


@dataclass
class TripletexAPIError(RuntimeError):
    method: str
    path: str
    status_code: int
    payload: Any

    @property
    def request_id(self) -> str | None:
        if isinstance(self.payload, dict):
            return self.payload.get("requestId")
        return None

    @property
    def searchable_text(self) -> str:
        if isinstance(self.payload, dict):
            parts = [
                str(self.payload.get("message", "")),
                str(self.payload.get("developerMessage", "")),
            ]
            for item in self.payload.get("validationMessages", []) or []:
                if isinstance(item, dict):
                    parts.append(str(item.get("field", "")))
                    parts.append(str(item.get("message", "")))
            return " | ".join(part for part in parts if part)
        return str(self.payload)

    def __str__(self) -> str:
        return f"{self.method} {self.path} -> {self.status_code}: {self.searchable_text}"


class TripletexClient:
    def __init__(self, base_url: str, session_token: str, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.auth = ("0", session_token)
        self.session.headers.update({"Content-Type": "application/json"})
        self.api_calls = 0
        self.api_errors = 0
        self._account_key_cache: str | None = None
        self._account_key_aliases_cache: list[str] | None = None
        self._session_identity_cache: dict[str, Any] | None = None
        self._forbidden_capabilities: set[tuple[str, str]] = set()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        method_upper = method.upper()
        normalized_path = "/" + path.lstrip("/")
        filtered_params = {key: value for key, value in (params or {}).items() if value is not None}
        contract_violations = validate_request_contract(
            method_upper,
            normalized_path,
            params=filtered_params,
            data=data,
        )
        if contract_violations:
            self.api_errors += 1
            payload = {
                "status": 460,
                "code": 9460,
                "message": "OpenAPI contract validation failed before sending request.",
                "developerMessage": f"{len(contract_violations)} contract rule(s) were violated.",
                "validationMessages": [
                    {
                        "field": violation.field,
                        "message": violation.message,
                        "rule": violation.rule,
                    }
                    for violation in contract_violations
                ],
            }
            logger.warning(
                "Tripletex contract error %s %s -> %s violation(s)",
                method_upper,
                normalized_path,
                len(contract_violations),
            )
            raise TripletexAPIError(method_upper, normalized_path, 460, payload)

        capability_key = self._capability_key(method_upper, normalized_path)
        if capability_key in self._forbidden_capabilities:
            raise TripletexAPIError(
                method_upper,
                normalized_path,
                403,
                {
                    "status": 403,
                    "code": 9000,
                    "message": "Skipped known forbidden endpoint for this account capability.",
                },
            )

        url = f"{self.base_url}/{path.lstrip('/')}"
        self.api_calls += 1

        response = self.session.request(
            method=method_upper,
            url=url,
            params=filtered_params or None,
            json=data,
            timeout=self.timeout_seconds,
        )

        if response.status_code >= 400:
            self.api_errors += 1
            payload = self._parse_response_body(response)
            logger.warning("Tripletex error %s %s -> %s", method_upper, normalized_path, response.status_code)
            if response.status_code == 403 and is_restricted_pilot_endpoint(method_upper, normalized_path):
                self._forbidden_capabilities.add(capability_key)
            raise TripletexAPIError(method_upper, normalized_path, response.status_code, payload)

        if response.status_code == 204 or not response.content:
            return {}

        payload = self._parse_response_body(response)
        if not isinstance(payload, dict):
            return {"value": payload}
        return payload

    @staticmethod
    def _capability_key(method: str, normalized_path: str) -> tuple[str, str]:
        path_key = re.sub(r"(?<=/)\d+(?=/|$)", "{id}", normalized_path)
        return method.upper(), path_key

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        data: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.request("POST", path, params=params, data=data)

    def put(
        self,
        path: str,
        *,
        data: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.request("PUT", path, params=params, data=data)

    def delete(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("DELETE", path, params=params)

    def get_stats(self) -> dict[str, int]:
        return {
            "total_calls": self.api_calls,
            "errors": self.api_errors,
        }

    def account_key(self) -> str:
        if self._account_key_cache is None:
            self._account_key_cache, self._account_key_aliases_cache = self._resolve_account_key()
        return self._account_key_cache

    def account_key_aliases(self) -> list[str]:
        if self._account_key_aliases_cache is None:
            self.account_key()
        return list(self._account_key_aliases_cache or [self.base_url])

    def session_identity(self) -> dict[str, Any]:
        if self._session_identity_cache is None:
            payload = self.request("GET", "/token/session/>whoAmI")
            value = payload.get("value") if isinstance(payload, dict) else None
            self._session_identity_cache = value if isinstance(value, dict) else {}
        return dict(self._session_identity_cache)

    @staticmethod
    def _parse_response_body(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text[:1000]

    def _resolve_account_key(self) -> tuple[str, list[str]]:
        aliases = [self.base_url]
        try:
            value = self.session_identity()
        except Exception as exc:
            logger.warning("Could not resolve Tripletex company identity, falling back to base URL: %s", exc)
            return self.base_url, aliases

        company_id = value.get("companyId")
        if company_id is not None:
            key = f"tripletex-company:{company_id}"
            if key not in aliases:
                aliases.insert(0, key)
            return key, aliases
        return self.base_url, aliases
