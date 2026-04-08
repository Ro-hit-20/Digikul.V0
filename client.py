"""
DigiKul-v0 — EnvClient for remote interaction.

Provides a typed client that connects to a running DigiKul
environment server via HTTP, implementing the OpenEnv client
pattern with proper payload serialization and result parsing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import httpx

from models import (
    DigiKulAction,
    DigiKulObservation,
    DigiKulState,
)


class DigiKulEnvClient:
    """
    HTTP client for the DigiKul-v0 environment.

    Usage (async):
        async with DigiKulEnvClient(base_url="http://localhost:7860") as client:
            obs = await client.reset()
            obs = await client.step(DigiKulAction(quality_levels=[3, 2, 1, 0, 1]))
            state = await client.state()

    Usage (sync):
        client = DigiKulEnvClient(base_url="http://localhost:7860")
        obs = client.reset_sync()
        obs = client.step_sync(DigiKulAction(quality_levels=[3, 2, 1, 0, 1]))
        state = client.state_sync()
        client.close_sync()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._async_client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None

    # ── Async context manager ──

    async def __aenter__(self):
        self._async_client = httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout
        )
        return self

    async def __aexit__(self, *args):
        if self._async_client:
            await self._async_client.aclose()

    # ── Async methods ──

    async def reset(self) -> DigiKulObservation:
        assert self._async_client, "Use 'async with' or create client first."
        resp = await self._async_client.post("/reset")
        resp.raise_for_status()
        return DigiKulObservation(**resp.json())

    async def step(self, action: DigiKulAction) -> DigiKulObservation:
        assert self._async_client, "Use 'async with' or create client first."
        resp = await self._async_client.post(
            "/step", json=action.model_dump()
        )
        resp.raise_for_status()
        return DigiKulObservation(**resp.json())

    async def state(self) -> DigiKulState:
        assert self._async_client, "Use 'async with' or create client first."
        resp = await self._async_client.get("/state")
        resp.raise_for_status()
        return DigiKulState(**resp.json())

    # ── Sync methods (convenience wrappers) ──

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self._base_url, timeout=self._timeout
            )
        return self._sync_client

    def reset_sync(self) -> DigiKulObservation:
        resp = self._get_sync_client().post("/reset")
        resp.raise_for_status()
        return DigiKulObservation(**resp.json())

    def step_sync(self, action: DigiKulAction) -> DigiKulObservation:
        resp = self._get_sync_client().post("/step", json=action.model_dump())
        resp.raise_for_status()
        return DigiKulObservation(**resp.json())

    def state_sync(self) -> DigiKulState:
        resp = self._get_sync_client().get("/state")
        resp.raise_for_status()
        return DigiKulState(**resp.json())

    def close_sync(self):
        if self._sync_client:
            self._sync_client.close()
