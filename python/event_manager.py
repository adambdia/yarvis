class Event_Manager():
    
    def __init__(self):
        self.events = {}
        
    def poll_event(self, event_name: str):
        if event_name not in self.events.keys(): raise IndexError('\"{}\" event does not exist'.format(event_name))
        return self.events[event_name]
    
    def push_event(self, event_name: str, value: any) -> None:
        if value is None: raise ValueError('Value passed to event cannot be None')
        if not event_name: raise ValueError('event_name cannot be an empty string')
        self.events[event_name] = value
    
    def view_event(self): print(self.events) # For DEBUGGING
