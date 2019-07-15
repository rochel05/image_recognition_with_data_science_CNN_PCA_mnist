import matplotlib.pyplot as plt
import numpy

f = open("datasets/mnist_test_10.csv", 'r')
a = f.readlines()
f.close()

f = plt.figure(figsize=(15,15));
count=1
for line in a:
    linebits = line.split(',')
    imarray = numpy.asfarray(linebits[1:]).reshape((28,28))
    plt.subplot(5,5,count)
    plt.subplots_adjust(hspace=0.5)
    count += 1
    plt.title("Label is " + linebits[0])
    plt.imshow(imarray, cmap='Greys', interpolation='None')
plt.show()