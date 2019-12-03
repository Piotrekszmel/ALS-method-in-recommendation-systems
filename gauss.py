import numpy as np
import random
import copy


class matrix:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.row = 1
        self.X = [1] * len(self.A)
        self.temp_A = copy.deepcopy(self.A)
        self.temp_B = copy.deepcopy(self.B)
        
    def show(self, X=False):
        for i in range(len(self.A)):
            wx = [j for j in self.A[i]]
            print(wx, " = ", self.B[i])
        print()

        if X == True:
            for i in range(len(self.X)):
                print("x_{} = ".format(i + 1), self.X[i])

    def Gauss(self):
        for i in range(self.row, len(self.A)):
            values = [(self.A[j][i], j) for j in range(i - 1, len(self.A))]
            max_value = max(value[0] for value in values)
            index_max = [val[1] for val in values if val[0] == max_value][0]
            
            B_temp = self.B[i - 1]
            self.B[i - 1] = self.B[index_max]
            self.B[index_max] = B_temp
                
            for j in range(len(self.A)):
                r = self.A[i - 1][j]
                self.A[i - 1][j] = self.A[index_max][j]
                self.A[index_max][j] = r
            
            for ii in range(i, len(self.A)):
                self.calculation_A_B(ii)

            self.row += 1
            
        self.X = self.count()
        
        return self.X

    def count(self):
        for row in range(len(self.A) -1, -1, -1):
            self.A[row][:row] = [A * x for A,x in zip(self.A[row][:row], self.X[:row])]
            self.A[row][row + 1:] = [A * x for A,x in zip(self.A[row][row + 1:], self.X[row + 1:])]
            
            values = np.float64(sum([-value for i, value in enumerate(self.A[row]) if i != row]))
            values += np.float64(self.B[row])
            self.X[row] = np.float64(values / self.A[row][row])
        
        self.A = self.temp_A
        self.B = self.temp_B

        return self.X
    
    def calculation_A_B(self, i):
        self.B[i] = np.float64(self.B[i] + (self.B[self.row - 1] * (-self.A[i][self.row - 1] / self.A[self.row - 1][self.row - 1])))
        temp = self.A[i][self.row - 1]
        for j in range(len(self.A)):
            self.A[i][j] = np.float64(self.A[i][j] + (self.A[self.row - 1][j] * (-temp / self.A[self.row - 1][self.row - 1])))