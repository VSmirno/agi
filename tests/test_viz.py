"""Tests for the visualization server (Stage 5).

Tests REST endpoints, WebSocket streaming, FPS gate, and end-to-end pipeline.
"""

from __future__ import annotations

import asyncio
import time

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from snks.daf.types import DafConfig, DcamConfig, EncoderConfig, PipelineConfig, SKSConfig
from snks.viz.server import app, reset_state, set_config, _run_one_cycle, _ensure_pipeline, _state

pytestmark = pytest.mark.asyncio(loop_scope="function")


def _tiny_pipeline_config() -> PipelineConfig:
    """Tiny config for fast viz tests."""
    return PipelineConfig(
        daf=DafConfig(
            num_nodes=500,
            avg_degree=10,
            dt=0.01,
            oscillator_model="fhn",
            fhn_I_base=0.5,
            device="cpu",
        ),
        encoder=EncoderConfig(image_size=64),
        dcam=DcamConfig(hac_dim=64, lsh_n_tables=4, lsh_n_bits=6, episodic_capacity=100),
        sks=SKSConfig(top_k=200, min_cluster_size=5),
        steps_per_cycle=30,
        device="cpu",
    )


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset server state and set tiny config before each test."""
    reset_state()
    set_config(_tiny_pipeline_config())
    yield
    reset_state()


@pytest_asyncio.fixture
async def client():
    """Async HTTP client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestStatusEndpoint:
    """Tests for GET /status."""

    @pytest.mark.asyncio
    async def test_status_returns_json(self, client: AsyncClient):
        resp = await client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "cycle_count" in data
        assert data["running"] is False
        assert data["cycle_count"] == 0

    @pytest.mark.asyncio
    async def test_status_reflects_step(self, client: AsyncClient):
        await client.post("/step")
        resp = await client.get("/status")
        data = resp.json()
        assert data["cycle_count"] == 1
        assert data["dcam_episodes"] >= 1


class TestStepEndpoint:
    """Tests for POST /step."""

    @pytest.mark.asyncio
    async def test_step_returns_cycle(self, client: AsyncClient):
        resp = await client.post("/step")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "cycle"
        assert "metrics" in data
        assert "top_nodes" in data
        assert "sks_clusters" in data

    @pytest.mark.asyncio
    async def test_step_increments_counter(self, client: AsyncClient):
        await client.post("/step")
        await client.post("/step")
        resp = await client.get("/status")
        assert resp.json()["cycle_count"] == 2

    @pytest.mark.asyncio
    async def test_step_metrics_structure(self, client: AsyncClient):
        resp = await client.post("/step")
        m = resp.json()["metrics"]
        assert "n_sks" in m
        assert "mean_pe" in m
        assert "n_spikes" in m
        assert "cycle_time_ms" in m
        assert "dcam_episodes" in m


class TestStartPause:
    """Tests for POST /start and POST /pause."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, client: AsyncClient):
        resp = await client.post("/start")
        assert resp.json()["status"] == "started"
        # Give the loop a moment to run
        await asyncio.sleep(0.5)
        status = (await client.get("/status")).json()
        assert status["running"] is True
        assert status["cycle_count"] >= 1
        # Cleanup
        await client.post("/pause")

    @pytest.mark.asyncio
    async def test_pause_stops_running(self, client: AsyncClient):
        await client.post("/start")
        await asyncio.sleep(0.3)
        resp = await client.post("/pause")
        assert resp.json()["status"] == "paused"
        status = (await client.get("/status")).json()
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_double_start(self, client: AsyncClient):
        await client.post("/start")
        resp = await client.post("/start")
        assert resp.json()["status"] == "already_running"
        await client.post("/pause")


class TestFPSGate:
    """Gate: Dashboard FPS >= 5."""

    @pytest.mark.asyncio
    async def test_fps_gate(self, client: AsyncClient):
        """Gate: FPS >= 5 over 20 cycles."""
        n_cycles = 20
        t0 = time.perf_counter()
        for _ in range(n_cycles):
            resp = await client.post("/step")
            assert resp.status_code == 200
        elapsed = time.perf_counter() - t0
        fps = n_cycles / elapsed
        print(f"\n  FPS Gate: {fps:.1f} FPS over {n_cycles} cycles ({elapsed:.2f}s)")
        assert fps >= 5, f"FPS {fps:.1f} < 5 required"


class TestEndToEndPipeline:
    """Gate: Full pipeline end-to-end with DCAM."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_dcam(self, client: AsyncClient):
        """Gate: End-to-end pipeline creates DCAM episodes."""
        # Run several cycles
        for _ in range(5):
            resp = await client.post("/step")
            assert resp.status_code == 200

        status = (await client.get("/status")).json()
        assert status["cycle_count"] == 5
        assert status["dcam_episodes"] >= 5, (
            f"Expected >= 5 DCAM episodes, got {status['dcam_episodes']}"
        )

    @pytest.mark.asyncio
    async def test_cycle_has_valid_data(self, client: AsyncClient):
        """Verify cycle data contains expected fields and types."""
        resp = await client.post("/step")
        data = resp.json()

        assert isinstance(data["cycle"], int)
        assert isinstance(data["top_nodes"], list)
        assert isinstance(data["sks_clusters"], dict)
        assert isinstance(data["metrics"]["n_spikes"], int)
        assert isinstance(data["metrics"]["mean_pe"], float)
        assert isinstance(data["metrics"]["cycle_time_ms"], float)


class TestIndexPage:
    """Tests for GET /."""

    @pytest.mark.asyncio
    async def test_index_serves_html(self, client: AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "СНКС" in resp.text or "text/html" in resp.headers.get("content-type", "")
