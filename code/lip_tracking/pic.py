import numpy as np
import matplotlib.pyplot as plt

err = np.load('/Users/zkx/Desktop/3dcnn/5_100_16/err.npy')
ap = np.load('/Users/zkx/Desktop/3dcnn/5_100_16/ap.npy')
auc = np.load('/Users/zkx/Desktop/3dcnn/5_100_16/auc.npy')
print(err.shape)
print(ap.shape)
print(auc.shape)
plt.figure()
plt.plot(auc)
plt.show()