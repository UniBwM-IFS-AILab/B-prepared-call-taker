import asyncio
import inspect


async def async_wrapper(value):
    return await value if inspect.isawaitable(value) else value


def sync_if_needed(value):
    if inspect.isawaitable(value):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(value)
    else:
        return value
