import sys, os, asyncio
from litellm.caching.in_memory_cache import InMemoryCache
import pytest
import copy
from unittest.mock import MagicMock, patch
from litellm.router_strategy.simple_shuffle import SimpleShuffleWithSessionsLoggingHandler, simple_shuffle
from litellm.caching.dual_cache import DualCache

@pytest.fixture
def healthy_deployments():
    return [
        {"model_name": "model-a", "litellm_params": {"weight": 1}},
        {"model_name": "model-b", "litellm_params": {"weight": 2}},
    ]

@pytest.fixture
def mock_router_instance():
    mock = MagicMock()
    mock.print_deployment = lambda deployment: deployment.get("model_name", str(deployment))
    return mock

@pytest.mark.asyncio
async def test_async_get_available_deployments_cache_miss(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    session_id = "abc"
    request_kwargs = {"metadata": {"session_id": session_id}}
    with patch("random.choices", return_value=[1]):
        deployment = await handler.async_get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-b"
    assert await cache.async_get_cache(f"thesis:session:{session_id}") == "model-b"

@pytest.mark.asyncio
async def test_async_get_available_deployments_cache_hit(healthy_deployments, mock_router_instance):
    cache = DualCache()
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    session_id = "abc"
    # Set cache to the second model in healthy_deployments
    await cache.async_set_cache(f"thesis:session:{session_id}", "model-b")
    request_kwargs = {"metadata": {"session_id": session_id}}
    deployment = await handler.async_get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=request_kwargs,
    )
    assert deployment["model_name"] == "model-b"

@pytest.mark.asyncio
async def test_async_get_available_deployments_cache_hit_but_model_missing(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    session_id = "abc"
    await cache.async_set_cache(f"thesis:session:{session_id}", "model-x")  # not in healthy_deployments
    request_kwargs = {"metadata": {"session_id": session_id}}
    with patch("random.choices", return_value=[0]):
        deployment = await handler.async_get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-a"
    assert await cache.async_get_cache(f"thesis:session:{session_id}") == "model-a"

@pytest.mark.asyncio
async def test_async_get_available_deployments_no_session_id(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    request_kwargs = {"metadata": {}}
    with patch("random.choices", return_value=[0]):
        deployment = await handler.async_get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-a"

@pytest.mark.asyncio
async def test_async_get_available_deployments_no_metadata(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    request_kwargs = {}
    with patch("random.choices", return_value=[1]):
        deployment = await handler.async_get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-b"

def test_get_available_deployments_cache_miss(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    session_id = "abc"
    request_kwargs = {"metadata": {"session_id": session_id}}
    with patch("random.choices", return_value=[1]):
        deployment = handler.get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-b"
    cache_result = cache.get_cache(f"thesis:session:{session_id}")
    print(f"\n cache_result is {cache_result}")
    assert cache_result == "model-b"

def test_get_available_deployments_cache_hit(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    session_id = "abc"
    # Set cache to the second model in healthy_deployments
    cache.set_cache(f"thesis:session:{session_id}", "model-b")
    request_kwargs = {"metadata": {"session_id": session_id}}
    deployment = handler.get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=request_kwargs,
    )
    assert deployment["model_name"] == "model-b"

def test_get_available_deployments_cache_hit_but_model_missing(healthy_deployments, mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    session_id = "abc"
    cache.set_cache(f"thesis:session:{session_id}", "model-x")  # not in healthy_deployments
    request_kwargs = {"metadata": {"session_id": session_id}}
    with patch("random.choices", return_value=[0]):
        deployment = handler.get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-a"
    assert cache.get_cache(f"thesis:session:{session_id}") == "model-a"

@pytest.mark.asyncio
async def test_async_empty_healthy_deployments(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    request_kwargs = {"metadata": {"session_id": "abc"}}
    with pytest.raises(IndexError):
        await handler.async_get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=[],
            model="model-a",
            request_kwargs=request_kwargs,
        )

def test_empty_healthy_deployments(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    request_kwargs = {"metadata": {"session_id": "abc"}}
    with pytest.raises(IndexError):
        handler.get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=[],
            model="model-a",
            request_kwargs=request_kwargs,
        )

@pytest.mark.asyncio
async def test_async_single_deployment(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [{"model_name": "model-a", "litellm_params": {"weight": 1}}]
    request_kwargs = {"metadata": {"session_id": "abc"}}
    deployment = await handler.async_get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=request_kwargs,
    )
    assert deployment["model_name"] == "model-a"

def test_single_deployment(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [{"model_name": "model-a", "litellm_params": {"weight": 1}}]
    request_kwargs = {"metadata": {"session_id": "abc"}}
    deployment = handler.get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=request_kwargs,
    )
    assert deployment["model_name"] == "model-a"

@pytest.mark.asyncio
async def test_async_all_same_model_name(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [
        {"model_name": "model-a", "litellm_params": {"weight": 1}},
        {"model_name": "model-a", "litellm_params": {"weight": 2}},
    ]
    request_kwargs = {"metadata": {"session_id": "abc"}}
    with patch("random.choices", return_value=[1]):
        deployment = await handler.async_get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-a"

def test_all_same_model_name(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [
        {"model_name": "model-a", "litellm_params": {"weight": 1}},
        {"model_name": "model-a", "litellm_params": {"weight": 2}},
    ]
    request_kwargs = {"metadata": {"session_id": "abc"}}
    with patch("random.choices", return_value=[1]):
        deployment = handler.get_available_deployments(
            llm_router_instance=mock_router_instance,
            healthy_deployments=healthy_deployments,
            model="model-a",
            request_kwargs=request_kwargs,
        )
    assert deployment["model_name"] == "model-a"

@pytest.mark.asyncio
async def test_async_missing_metadata_key(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [{"model_name": "model-a", "litellm_params": {"weight": 1}}]
    request_kwargs = {}  # no 'metadata' key
    deployment = await handler.async_get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=request_kwargs,
    )
    assert deployment["model_name"] == "model-a"

def test_missing_metadata_key(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [{"model_name": "model-a", "litellm_params": {"weight": 1}}]
    request_kwargs = {}  # no 'metadata' key
    deployment = handler.get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=request_kwargs,
    )
    assert deployment["model_name"] == "model-a"

@pytest.mark.asyncio
async def test_async_request_kwargs_none(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [{"model_name": "model-a", "litellm_params": {"weight": 1}}]
    deployment = await handler.async_get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=None,
    )
    assert deployment["model_name"] == "model-a"

def test_request_kwargs_none(mock_router_instance):
    cache = DualCache(in_memory_cache=InMemoryCache())
    handler = SimpleShuffleWithSessionsLoggingHandler(cache)
    healthy_deployments = [{"model_name": "model-a", "litellm_params": {"weight": 1}}]
    deployment = handler.get_available_deployments(
        llm_router_instance=mock_router_instance,
        healthy_deployments=healthy_deployments,
        model="model-a",
        request_kwargs=None,
    )
    assert deployment["model_name"] == "model-a" 