import numpy as np

class ArmModel():
    def __init__(self):
        self.arm_points = [2, 3, 4]  #V1, V2, V3

    def __call__(self, points):
        vector = np.zeros([3,2], dtype='float32')
        state = 0
        for i in range(0, 3):
            vector[i][0] = points[self.arm_points[i]][0]
            vector[i][1] = points[self.arm_points[i]][1]
        if sum(vector[2]) and sum(vector[0]) != 0:
            Lx = [vector[0][0] - vector[1][0], vector[0][1] - vector[1][1]]
            Ly = [vector[2][0] - vector[1][0], vector[2][1] - vector[1][1]]
            if Lx[0] > 0 and Lx[1] < 0 and Ly[0] < 0 and Ly[1] < 0 :
                return state
            else:
                state = 1
                return state
        else:
            state = 1
            return state


