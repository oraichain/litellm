"""
Returns a random deployment from the list of healthy deployments.

If weights are provided, it will return a deployment based on the weights.

"""

import random
import copy
from typing import TYPE_CHECKING, Any, Dict, List, Union

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

class SimpleShuffleWithSessionsLoggingHandler(CustomLogger):
    SESSION_KEY_PREFIX = "thesis:session:"
    # NOTE: Why 3 hours?
    SESSION_IN_MEMORY_CACHE_TTL = 60 * 60 * 3
    SESSION_REDIS_CACHE_TTL = 60 * 60 * 3
    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        self.dual_cache.update_cache_ttl(self.SESSION_IN_MEMORY_CACHE_TTL, self.SESSION_REDIS_CACHE_TTL)

    async def async_get_available_deployments(self, llm_router_instance: LitellmRouter, healthy_deployments: Union[List[Any], Dict[Any, Any]], model: str, request_kwargs: Dict[Any, Any] | None = None) -> Dict:
        if request_kwargs is None:
            return simple_shuffle(llm_router_instance, healthy_deployments, model)
        metadata = request_kwargs.get("metadata", {})
        verbose_router_logger.debug(f"\n metadata {metadata}")
        if metadata is None:
            return simple_shuffle(llm_router_instance, healthy_deployments, model)
        session_id = metadata.get("session_id", None)
        if session_id is None:
            return simple_shuffle(llm_router_instance, healthy_deployments, model)
        verbose_router_logger.info(f"\n session_id: {session_id}")
        deployment_name_cache = await self.dual_cache.async_get_cache(f"{self.SESSION_KEY_PREFIX}{session_id}")
        if not deployment_name_cache:
            deployment = simple_shuffle(llm_router_instance, healthy_deployments, model)
            model_name = deployment.get("model_name", '')
            await self.dual_cache.async_set_cache(f"{self.SESSION_KEY_PREFIX}{session_id}", model_name)
            return deployment
        for deployment in healthy_deployments:
            if deployment.get("model_name") == deployment_name_cache:
                return deployment
        # If no matching deployment found, fallback to simple shuffle and update cache
        deployment = simple_shuffle(llm_router_instance, healthy_deployments, model)
        model_name = deployment.get("model_name", '')
        await self.dual_cache.async_set_cache(f"{self.SESSION_KEY_PREFIX}{session_id}", model_name)
        # Ensure cache is updated before returning
        return deployment

    def get_available_deployments(self, llm_router_instance: LitellmRouter, healthy_deployments: Union[List[Any], Dict[Any, Any]], model: str, request_kwargs: Dict[Any, Any] | None = None) -> Dict:
        if request_kwargs is None:
            return simple_shuffle(llm_router_instance, healthy_deployments, model)
        metadata = request_kwargs.get("metadata", {})
        verbose_router_logger.debug(f"\n metadata {metadata}")
        if metadata is None:
            return simple_shuffle(llm_router_instance, healthy_deployments, model)
        session_id = metadata.get("session_id", None)
        if session_id is None:
            return simple_shuffle(llm_router_instance, healthy_deployments, model)
        verbose_router_logger.debug(f"\n session_id {session_id}")
        deployment_name_cache = self.dual_cache.get_cache(f"{self.SESSION_KEY_PREFIX}{session_id}")
        if not deployment_name_cache:
            deployment = simple_shuffle(llm_router_instance, healthy_deployments, model)
            model_name = deployment.get("model_name", '')
            self.dual_cache.set_cache(f"{self.SESSION_KEY_PREFIX}{session_id}", model_name)
            return deployment
        for deployment in healthy_deployments:
            if deployment.get("model_name") == deployment_name_cache:
                return deployment
        # If no matching deployment found, fallback to simple shuffle and update cache
        deployment = simple_shuffle(llm_router_instance, healthy_deployments, model)
        model_name = deployment.get("model_name", '')
        self.dual_cache.set_cache(f"{self.SESSION_KEY_PREFIX}{session_id}", model_name)
        # Ensure cache is updated before returning
        return deployment

def simple_shuffle(
    llm_router_instance: LitellmRouter,
    healthy_deployments: Union[List[Any], Dict[Any, Any]],
    model: str,
) -> Dict:
    """
    Returns a random deployment from the list of healthy deployments.

    If weights are provided, it will return a deployment based on the weights.

    If users pass `rpm` or `tpm`, we do a random weighted pick - based on `rpm`/`tpm`.

    Args:
        llm_router_instance: LitellmRouter instance
        healthy_deployments: List of healthy deployments
        model: Model name

    Returns:
        Dict: A single healthy deployment
    """

    ############## Check if 'weight' param set for a weighted pick #################
    weight = healthy_deployments[0].get("litellm_params").get("weight", None)
    if weight is not None:
        # use weight-random pick if rpms provided
        weights = [m["litellm_params"].get("weight", 0) for m in healthy_deployments]
        verbose_router_logger.debug(f"\nweight {weights}")
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]
        verbose_router_logger.debug(f"\n weights {weights}")
        # Perform weighted random pick
        selected_index = random.choices(range(len(weights)), weights=weights)[0]
        verbose_router_logger.debug(f"\n selected index, {selected_index}")
        deployment = healthy_deployments[selected_index]
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {llm_router_instance.print_deployment(deployment) or deployment[0]} for model: {model}"
        )
        return deployment or deployment[0]
    ############## Check if we can do a RPM/TPM based weighted pick #################
    rpm = healthy_deployments[0].get("litellm_params").get("rpm", None)
    if rpm is not None:
        # use weight-random pick if rpms provided
        rpms = [m["litellm_params"].get("rpm", 0) for m in healthy_deployments]
        verbose_router_logger.debug(f"\nrpms {rpms}")
        total_rpm = sum(rpms)
        weights = [rpm / total_rpm for rpm in rpms]
        verbose_router_logger.debug(f"\n weights {weights}")
        # Perform weighted random pick
        selected_index = random.choices(range(len(rpms)), weights=weights)[0]
        verbose_router_logger.debug(f"\n selected index, {selected_index}")
        deployment = healthy_deployments[selected_index]
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {llm_router_instance.print_deployment(deployment) or deployment[0]} for model: {model}"
        )
        return deployment or deployment[0]
    ############## Check if we can do a RPM/TPM based weighted pick #################
    tpm = healthy_deployments[0].get("litellm_params").get("tpm", None)
    if tpm is not None:
        # use weight-random pick if rpms provided
        tpms = [m["litellm_params"].get("tpm", 0) for m in healthy_deployments]
        verbose_router_logger.debug(f"\ntpms {tpms}")
        total_tpm = sum(tpms)
        weights = [tpm / total_tpm for tpm in tpms]
        verbose_router_logger.debug(f"\n weights {weights}")
        # Perform weighted random pick
        selected_index = random.choices(range(len(tpms)), weights=weights)[0]
        verbose_router_logger.debug(f"\n selected index, {selected_index}")
        deployment = healthy_deployments[selected_index]
        verbose_router_logger.info(
            f"get_available_deployment for model: {model}, Selected deployment: {llm_router_instance.print_deployment(deployment) or deployment[0]} for model: {model}"
        )
        return deployment or deployment[0]

    ############## No RPM/TPM passed, we do a random pick #################
    item = random.choice(healthy_deployments)
    return item or item[0]
