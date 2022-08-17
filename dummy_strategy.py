class DummyStrategyScope:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass


class DummyStrategy:
    def __init__(self):
        self.num_replicas_in_sync = 1
    def scope(self):
        return DummyStrategyScope()
    def experimental_distribute_dataset(self, dataset, *args):
        return dataset
    def run(self, f, args=()):
        return f(*args)
    def reduce(self, mode, value, axis=None):
        return value
    def gather(self, t, axis):
        return t