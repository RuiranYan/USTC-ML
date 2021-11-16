import numpy as np
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def pr():
    print('hello world')
    return 1

y = np.array([[0],[1],[2],[2],[1]])
print(convert_to_one_hot(y,3))
print('s' + str(pr()))