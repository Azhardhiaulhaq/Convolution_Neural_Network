class Layer:
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return self.call(*args)
    
    def call(self, input_data):
        return input_data
    
    def back_propagation(self, error, momentum):
        return error

    # Update pass for non weighted layer
    def update(self, learning_rate):
        pass