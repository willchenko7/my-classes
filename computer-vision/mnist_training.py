from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

single_image = X_train[0]
import matplotlib.pyplot as plt
#save img to tmp file
plt.imsave('tmp/tmp63.jpg', single_image, cmap='gray_r')