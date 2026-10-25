class AttrDict(dict):
    def __init__(self, value):
        super().__init__({k: AttrDict(v) if isinstance(v, dict) else v for k,v in value.items()})
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as e: raise AttributeError(key) from e
