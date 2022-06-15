import multiprocessing as mp
from typing import Callable
import signal


class ConcurrentDataLoader:

    def __init__(self, get_batch: Callable, queue_len: int = 1):
        self._get_batch = get_batch
        self.queue = mp.Queue(maxsize=queue_len)
        self.proc = mp.Process(target=self._prepare_batch, args=(get_batch, self.queue))
        self.proc_running = mp.Event()
        self.proc_running.set()
        self.proc.start()

        # exit gracefully
        def expanded_handler(signum, frame):
            self._kill_proc()
            signal.default_int_handler(signum, frame)
        signal.signal(signal.SIGINT, expanded_handler)

    def _kill_proc(self):
        self.proc_running.clear()
        self.proc.terminate()
        self.proc.join()

    def __del__(self):
        self._kill_proc()

    def _prepare_batch(self, get_batch_fn: Callable, queue: mp.Queue):
        while self.proc_running.is_set():
            s, a, r, done = get_batch_fn()
            queue.put((s, a, r, done), block=True, timeout=None)

    def get_batch(self):
        return self.queue.get(block=True, timeout=None)