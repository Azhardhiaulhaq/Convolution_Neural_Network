from Layer import Layer

class Detector(Layer) :
    def __init__(self,activation="relu"):
        super().__init__()
        self.activation = activation

    def relu(self,input):
        for i in range(len(input)):
            for j in range(len(input[0])):
                for k in range(len(input[0][0])):
                    if input[i][j][k] < 0:
                        input[i][j][k] = 0  
        return input

    def call(self, input):
        if(self.activation == "relu"):
            return self.relu(input)
        else:
            raise ValueError("Method {} unsupported".format(self.mode))
    
    def back_propagation(self,error):
        error[error < 0.] = 0.
        return error