import httpx


async def get_httpx_client():
    timeout = httpx.Timeout(timeout=240)
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        yield client
