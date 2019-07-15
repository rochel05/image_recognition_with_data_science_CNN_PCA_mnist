from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import listdir
import math
from keras.preprocessing.image import load_img, img_to_array
from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

def getIndex(YList):
    index = 1
    #print("YList : ", YList)
    for data in YList:
        #print("data : ", data)
        if data == 1.0:
            retour = index
        else:
            index += 1
    return retour

# define image extractor function
def image__features_extractor(directory):
    for img in listdir(directory):
        print(img)
        filename = directory + '/' + img
        image = load_img(filename, target_size=(28, 28))
        image = img_to_array(image)
    return image

def classifyOne():
    # define categorical label
    label_dict = {
        0: 'Zero',
        1: 'Un',
        2: 'Deux',
        3: 'Trois',
        4: 'Quatre',
        5: 'Cinque',
        6: 'Six',
        7: 'Sept',
        8: 'Huit',
        9: 'Neuf',
    }

    #load heterogenous data
    Xtrain1, Xtest1 = imageLoader()
    Ytrain1, Ytest1 = sentenceLoader()
    Xtest2, Ytest2, Xtrain2, Ytrain2 = load_data_from_csv()
    Xtest3, Ytest3, Xtrain3, Ytrain3 = load_data_from_mongoDb()

    #agregate data with numpy
    Xtrain, Ytrain, Xtest, Ytest = agregation_of_heterogenous_datas(Xtrain1, Ytrain1, Xtrain2, Ytrain2, Xtrain3, Ytrain3, Xtest1, Ytest1, Xtest2, Ytest2, Xtest3, Ytest3)
    #reduce dimension of agregated data with PCA
    Xtrain, Xtest = reduction_of_dimension_with_PCA(Xtrain, Xtest)

    # data preprocessing
    # reshape
    Xtrain = Xtrain.reshape(-1, 28, 28, 1)
    Xtest = Xtest.reshape(-1, 28, 28, 1)
    print('Xtrain shape : {}'.format(Xtrain.shape))
    print('Xtest shape : {}'.format(Xtest.shape))

    # rescale image
    #Xtrain = Xtrain.astype('float32')
    #Xtest = Xtest.astype('float32')
    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    # rescale and reshape image
    input_image = 'image_to_recognize'
    image = image__features_extractor(input_image)
    image = np.resize(image, (1,28, 28, 1))
    image = np.array(image)/255

    print('image rescaled : {}'.format(image))
    print('input image shape : {}'.format(image.shape))

    #call model
    classifierCNN = load_model('model/classifier_cnn_pca_raw_csv_hetero_epoch10.h5py')
    #print(classifierCNN.summary())

    # predict label category rely for the input image
    predict = classifierCNN.predict(image)
    print("predicted : ", predict)
    predict = np.argmax(np.round(predict), axis=1)
    print("predicted : ", predict)
    predict = to_categorical(predict, num_classes=10)
    print("predicted : ", predict)
    print('predict class shape : {}'.format(predict.shape))
    print('right class shape : {}'.format(Ytest.shape))

    image = np.resize(image, (1,784))
    image = np.array(image)/255

    # show result
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.imshow(image.reshape(28, 28), cmap='gray_r')
    plt.suptitle('predict info : {}'.format(getIndex(predict[0])), fontsize=16)
    plt.show()

if __name__=="__main__":
    #classify()
    classifyOne()

"""
def classify():
    # define categorical label
    label_dict = {
        0: 'Zero',
        1: 'Un',
        2: 'Deux',
        3: 'Trois',
        4: 'Quatre',
        5: 'Cinque',
        6: 'Six',
        7: 'Sept',
        8: 'Huit',
        9: 'Neuf',
    }
    #load data
    #load heterogenous data
    Xtrain1, Xtest1 = imageLoader()
    Ytrain1, Ytest1 = sentenceLoader()
    Xtest2, Ytest2, Xtrain2, Ytrain2 = load_data_from_csv()
    #df_train_x, df_train_y = load_data_from_mongoDb()
    #Xtrain3, Xtest3, Ytrain3, Ytest3 = train_test_split(df_train_x, df_train_y, random_state=0, test_size=0.2)

    #combine data
    Xtrain = np.append(Xtrain1, Xtrain2, axis=0)
    Xtest = np.append(Xtest1, Xtest2, axis=0)
    Ytrain = np.append(Ytrain1, Ytrain2, axis=0)
    Ytest = np.append(Ytest1, Ytest2, axis=0)
    print(' Xtrain1 shape : {} - Ytrain1 shape : {}'.format(Xtrain1.shape, Ytrain1.shape))
    print(' Xtest1 shape : {} - Ytest1 shape : {}'.format(Xtest1.shape, Ytest1.shape))
    print(' Xtrain2 shape : {} - Ytrain2 shape : {}'.format(Xtrain2.shape, Ytrain2.shape))
    print(' Xtest2 shape : {} - Ytest2 shape : {}'.format(Xtest2.shape, Ytest2.shape))

    # data preprocessing
    # reshape
    Xtrain = Xtrain.reshape(-1, 28, 28, 1)
    Xtest = Xtest.reshape(-1, 28, 28, 1)
    print(' Xtrain shape : {} - Ytrain shape : {}'.format(Xtrain.shape, Ytrain.shape))
    print(' Xtest shape : {} - Ytest shape : {}'.format(Xtest.shape, Ytest.shape))

    # rescale image
    #Xtrain = Xtrain.astype('float32')
    #Xtest = Xtest.astype('float32')
    Xtrain = Xtrain / 255
    Xtest = Xtest / 255

    classifierCNN = load_model('classifier_cnn_pca_raw_csv_hetero_epoch10.h5py')
    predicted = classifierCNN.predict(Xtest1)
    predicted = np.argmax(np.round(predicted), axis=1)
    predicted = to_categorical(predicted, num_classes=10)
    print('predicted class shape : {}'.format(predicted.shape))
    print('right class shape : {}'.format(Ytest.shape))

    plt.figure(figsize=(20, 5))
    for i in range(10):
        #index = Ytest.tolist().value(i)
        plt.subplot(2, 10, i + 1)
        plt.imshow(Xtest[i].reshape((28, 28)), cmap='gray_r')
        plt.title("Class : {}".format(label_dict[getIndex(Ytest[i])]))
        plt.tight_layout()
    plt.show()
"""