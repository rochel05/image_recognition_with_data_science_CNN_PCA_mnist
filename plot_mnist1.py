import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='datasets/')

# Convert sklearn 'datasets bunch' object to Pandas DataFrames
y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)
print(X.head())
print(X.shape, y.shape)

first_image = X.loc[0,:]
first_label = y[0]
plt.imshow(np.reshape(first_image.values, (28, 28)), cmap='gray_r')
plt.title('Digit Label: {}'.format(first_label))
plt.show()

# Change column-names in X to reflect that they are pixel values
#num_images = X.shape[1]
#X.columns = ['pixel_'+str(x) for x in range(num_images)]

#X_values = pd.Series(X.values.ravel())
#print(" min: {}, \n max: {}, \n mean: {}, \n median: {}, \n most common value: {}".format(X_values.min(), X_values.max(),X_values.mean(), X_values.median(), X_values.value_counts().idxmax()))

#print(len(np.unique(X.values)))
