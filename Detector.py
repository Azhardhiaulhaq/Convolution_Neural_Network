from Layer import Layer

class Detector(Layer) :
    def __init__(self):
        super().__init__()

    def relu(self,input):
        for i in range(len(input)):
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = 0  
        return input
    
    def call(self,input):
        return self.relu(input)

# mat = [[1,1,-1],[2,-2,2],[-3,3,3]]
# detect = Detector()
# result = detect.relu(mat)
# print(result)
