import logging
from typing import Any

from httpx import AsyncClient, HTTPStatusError, TransportError
from pydantic_ai.providers import (
    Provider,
    gateway,
    google,
    google_cloud,
    infer_provider_class,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import (
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from arox.core.session import AgentRunInfo

logger = logging.getLogger(__name__)


def create_retrying_client(extra_request_hooks=None, **client_args):
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 499, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    async def log_request(request):
        logger.info(f"Sending request: {request.method} {request.url}")

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type(
                (HTTPStatusError, TransportError, ConnectionError)
            ),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=2, max=30)
            ),
            stop=stop_after_attempt(8),
            # Re-raise the last exception if all retries fail
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ),
        validate_response=should_retry_status,
    )
    request_hooks = [log_request] + (extra_request_hooks or [])
    return AsyncClient(
        transport=transport,
        event_hooks={"request": request_hooks},
        **client_args,
    )


# Copyied from pydantic_ai.providers.infer_provider and add http_client parameter.
def infer_provider(
    provider: str,
    run_info: AgentRunInfo,
    base_url: str = "",
    session_header: str = "X-Session-Id",
    turn_header: str = "X-Turn-Id",
) -> Provider[Any]:
    """Infer the provider from the provider name."""

    async def _add_context_headers(request):
        session_id = run_info.llm_context_id
        turn_id = run_info.run_id
        if session_id and session_header:
            request.headers[session_header] = session_id
        if turn_id and turn_header:
            request.headers[turn_header] = turn_id

    client = create_retrying_client(
        extra_request_hooks=[_add_context_headers],
    )

    kwargs: dict[str, Any] = {"http_client": client}
    if base_url:
        kwargs["base_url"] = base_url

    if provider.startswith("gateway/"):
        upstream_provider = provider.removeprefix("gateway/")
        return gateway.gateway_provider(upstream_provider, **kwargs)
    elif provider in ("google", "google-cloud"):
        # Google GenAI SDK uses HttpOptions.timeout for both the httpx
        # per-request timeout AND the X-Server-Timeout header sent to the
        # server. pydantic_ai reads the httpx client's timeout and forwards
        # it to HttpOptions.timeout, so they are always coupled.
        #
        # To decouple them we:
        # 1. Set timeout to 40, which is set for both client and server timeout by genai sdk.
        # 2. Then use an httpx request event hook to remove the X-Server-Timeout
        #    header before the request is sent, so the server is not
        #    constrained by that deadline.
        async def _remove_server_timeout(request):
            request.headers.pop("X-Server-Timeout", None)

        client = create_retrying_client(
            timeout=80,
            extra_request_hooks=[_remove_server_timeout, _add_context_headers],
        )
        kwargs["http_client"] = client
        if provider == "google-cloud":
            return google_cloud.GoogleCloudProvider(**kwargs)
        return google.GoogleProvider(**kwargs)
    else:
        provider_class = infer_provider_class(provider)
        return provider_class(**kwargs)
