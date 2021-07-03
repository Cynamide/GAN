import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1]]).reshape(-1,2)
b = np.array([0,1,1,1,1,1,0,0,1]).reshape(-1,1)
for i in range(len(b)):
    a[i][b[i]] = 0

print(a)