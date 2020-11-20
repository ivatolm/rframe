class Callback:
    def __init__(self, target):
        self.target = target

    def call(self, data):
        self.target(data)
