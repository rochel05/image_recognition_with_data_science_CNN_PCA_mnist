from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from load_data_from_csv import load_data_from_csv
from load_data_from_mongoDb import load_data_from_mongoDb
from load_data_from_rawData_Img import imageLoader
from load_data_from_rawData_txt import sentenceLoader
from keras.utils import to_categorical
from model import CnnClassifierModel
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .model import agregation_of_heterogenous_datas, reduction_of_dimension_with_PCA, reduction_of_dimension_with_LDA

def train():
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

    #define CNN architecture
    classifierCNN = CnnClassifierModel()

    #fit data into model
    classifierCNN_train = classifierCNN.fit(Xtrain, Ytrain, epochs=10, batch_size=64, validation_data=(Xtest, Ytest))
    classifierCNN.save('model/classifier_cnn_pca_raw_csv_hetero_epoch10.h5py')
    accuracy = classifierCNN_train.history['acc']
    accuracy_eval = classifierCNN_train.history['val_acc']
    loss = classifierCNN_train.history['loss']
    loss_eval = classifierCNN_train.history['val_loss']

    epochs = range(len(accuracy))
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.plot(epochs, accuracy, 'bo', label="Training accuracy")
    plt.plot(epochs, accuracy_eval, 'b', label="Validation Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(122)
    plt.plot(loss, 'bo', label="Training loss")
    plt.plot(epochs, loss_eval, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    print('the training of the CNN_classifier_model has been finished!!!...')

if __name__=="__main__":
    train()
    #image_classification_with_CNN_and_mongoDb_bigData