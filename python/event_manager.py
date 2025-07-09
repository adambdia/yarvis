import threading


class Event_Manager:
    def __init__(self):
        self._lock = threading.Lock()
        self.events = {}

    def poll_event(self, event_name: str):
        with self._lock:
            return self.events.get(event_name, False)

    def push_event(self, event_name: str, value: any) -> None:
        with self._lock:
            if value is None:
                raise ValueError("Value passed to event cannot be None")
            if not event_name:
                raise ValueError("event_name cannot be an empty string")
            self.events[event_name] = value

    def view_event(self):
        print(self.events)  # For DEBUGGING
