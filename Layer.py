class Layer:
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return self.call(*args)
    
    def call(self, input_data):
        return input_data
    
    # Update pass for non weighted layer
    def update(self, learning_rate):
        pass