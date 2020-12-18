import numpy as np

a =np.array([[1,2],[2,3]])
b =np.array([[1,2],[2,3]])
# d=np.array([,])
# d=np.ndarray([], dtype=np.int64).reshape(2,2)
d = np.ndarray(shape=(2,2), dtype=int)
c=np.concatenate([a,d])
print(c)