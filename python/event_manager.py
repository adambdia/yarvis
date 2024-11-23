class Event_Manager():
    
    def __init__(self):
        self.events = {}
        
    def read_event(self, event_name: str):
        if not str: raise ValueError('event_name cannot be an empty string')
        return self.events[event_name]
    
    def write_event(self, event_name: str, value: any) -> None:
        if value is None: raise ValueError('Value passed to event cannot be None')
        self.events[event_name] = value
        
