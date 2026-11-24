"""Owned loopback TCP forwarding to one already verified private container IP.

No HTTP rewriting, retries, host-network container, or daemon configuration.
The caller must establish container/network ownership before constructing this.
"""

import asyncio
import ipaddress
import threading


class LoopbackForward:
    def __init__(self, host, port, *, listen_port=0):
        address = ipaddress.ip_address(host)
        if address.version != 4 or not address.is_private or address.is_unspecified:
            raise ValueError("forward destination must be a verified private IPv4 address")
        if any(type(p) is not int or not 0 <= p <= 65535 for p in (port, listen_port)) or port == 0:
            raise ValueError("invalid TCP forward ports")
        self.host, self.port, self.listen_port = str(address), port, listen_port
        self._ready, self._done = threading.Event(), threading.Event()
        self._loop = self._stop = None
        self._error = None
        self._tasks = set()
        self._thread = threading.Thread(target=self._main, daemon=True, name="owned-loopback-forward")
        self._thread.start()
        if not self._ready.wait(5) or self._error:
            self.close()
            raise RuntimeError("owned_loopback_forward_start_failed")

    def _main(self):
        try:
            asyncio.run(self._serve())
        except BaseException:
            self._error = "owned_loopback_forward_failed"
        finally:
            self._done.set()
            self._ready.set()

    async def _serve(self):
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        server = await asyncio.start_server(self._accept, "127.0.0.1", self.listen_port)
        self.listen_port = server.sockets[0].getsockname()[1]
        self._ready.set()
        try:
            await self._stop.wait()
        finally:
            server.close()
            for task in tuple(self._tasks):
                task.cancel()
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
            self._tasks.clear()
            # Python 3.12 waits for accepted transports too. Cancel handlers
            # first so idle sockets cannot deadlock server.wait_closed().
            await server.wait_closed()

    async def _accept(self, downstream_read, downstream_write):
        task = asyncio.current_task()
        if len(self._tasks) >= 32 or self._stop.is_set():
            downstream_write.close()
            return
        self._tasks.add(task)
        upstream_write = None

        async def pump(reader, writer):
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()

        try:
            async with asyncio.timeout(120):
                upstream_read, upstream_write = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), timeout=5,
                )
                await asyncio.gather(pump(downstream_read, upstream_write),
                                     pump(upstream_read, downstream_write))
        except (OSError, TimeoutError):
            pass  # Connection failure is visible to the client; never replay input.
        finally:
            # Close both transports synchronously before any cancellable wait.
            # Shutdown may interrupt a handler already waiting for peer EOF.
            for writer in (downstream_write, upstream_write):
                if writer is not None:
                    writer.close()
            self._tasks.discard(task)
            for writer in (downstream_write, upstream_write):
                if writer is not None:
                    try:
                        await asyncio.wait_for(writer.wait_closed(), timeout=2)
                    except (OSError, TimeoutError):
                        pass

    def close(self):
        if self._loop is not None and self._stop is not None and not self._done.is_set():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("owned_loopback_forward_cleanup_unconfirmed")
