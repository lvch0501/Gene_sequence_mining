import math
import numpy as np
batch_size = 200
with open('../test_dataset/x.txt') as file:
    data_x = file.read()
    data_x = np.array(eval(data_x))
length = len(data_x)
for i in range(0, length, batch_size):
    j = np.array(data_x[i:min(length, i+batch_size)])
    print(str(i*batch_size)+":"+str((i+1)*batch_size))