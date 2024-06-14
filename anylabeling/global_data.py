import threading


class DictManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {}
        self._listeners = []

    def __setitem__(self, key, value):
        with self._lock:
            prevValue = self._data.get(key)
            self._data[key] = value
        if prevValue != value:
            self._notify_listeners(key, value, prevValue)

    def __getitem__(self, key):
        with self._lock:
            return self._data.get(key)

    def listen(self, key, listener):
        with self._lock:
            self._listeners.append((key, listener))

    def unlisten(self, key, listener):
        with self._lock:
            self._listeners.remove((key, listener))

    def _notify_listeners(self, key, nextValue, prevValue=None):
        for k, listener in self._listeners:
            if k == key:
                listener(nextValue, prevValue)
            elif k == "*":
                listener(key, nextValue, prevValue)


GlobalData = DictManager()
