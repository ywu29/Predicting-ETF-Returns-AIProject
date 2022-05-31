

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
from datetime import datetime
import itertools
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from random import randint
from sklearn.metrics import confusion_matrix
import tensorflow
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Input
import matplotlib.pyplot as plt


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #list of all stocks
ETFlist = ["BIL","BND","DBC","EEM","EFA","EMB","GLD","HYG","IJR","LQD","PFF","QQQ","SPY","TIP","TLT","VGK","VNQ","VPL","VT","VTI"]
    #number of features
data_size = 32
    #train-test split
split_perc = .8

returnlabels = [] #set empty
for i in range(12): #for 12 months
    returnlabels.append(f"tminus{2+i}") #add column label for each monthly return
for i in range(20): #for 20 days
    returnlabels.append(f"day{i+1}") #add column label for each daily return

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# define model
model = Sequential()
#Input lstm layer
model.add(LSTM(32, activation='relu', input_shape=(None, 32), return_sequences=True))
#second long short term memory layer
model.add(LSTM(units=40, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False, return_sequences=True))
#third long short term memory layer
model.add(LSTM(units=4, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False, return_sequences=True))
#fourth long short term memory layer
model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.0, recurrent_activation="sigmoid", activation="tanh",use_bias=True, unroll=False))
#model.add(Flatten()) #flatten data
#model.add(Dense(50, activation='relu')) #add dense layer
#final classification output layer
model.add(Dense(1, activation='sigmoid'))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model/Output
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

#model KPI table
KPI = pd.DataFrame(columns = ['Training Time', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss'])

    #for each stock
for stock in ETFlist:
    data = pd.read_csv(f"{stock}_normalized_features.csv") #import normalized data
    labels = data["nextreturn"] #separate into labels
    features = data[returnlabels] #and features
    split = round(len(labels)*split_perc) #must split on index rather than random sampling because time series is ordered data
    x_train = features[:split].to_numpy()
    y_train = labels[:split].to_numpy()
    x_test = features[split:].to_numpy()
    y_test = labels[split:].to_numpy()
    TSG_train = TimeseriesGenerator( #create training time series generator
        x_train,
        y_train,
        1,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size = 1
    )
    TSG_test = TimeseriesGenerator( #create validation time series generator
        x_test,
        y_test,
        1,
        sampling_rate=1,
        stride=1,
        start_index=0,
        end_index=None,
        shuffle=False,
        reverse=False,
        batch_size = 1
    )
    
    start_time = datetime.now() #start clock
    
    # fit the model
    history = model.fit(TSG_train, epochs=70, verbose=1, validation_data=(TSG_test)) #save history
    
    model.save(f'model.{stock}') #save unique model for each stock
           

    loss, accuracy = model.evaluate(TSG_test, verbose=0) #output model accuracy and loss
    print("")
    print('Accuracy: %f' % (accuracy*100)) #print accuracy
    
    stop_time = datetime.now() #stop clock
    run_time = stop_time - start_time
    print ("Time required for training:",run_time, "\n") #print total run time
    
    acc = history.history['accuracy'] #accuracy per epoch
    val_acc = history.history['val_accuracy'] #validation accuracy per epoch
    loss = history.history['loss'] #loss per epoch
    val_loss = history.history['val_loss'] #validation loss per epoch
    
    epochs = range(len(acc)) #total number of epochs
    
    KPI.loc[stock] = [run_time, acc[-1], val_acc[-1], loss[-1], val_loss[-1]]
    
    fig,ax = plt.subplots() #create plot
    plt.plot(epochs, acc, 'blue', label='Training acc') #plot training accuracy line
    plt.plot(epochs, val_acc, 'red', label='Validation acc') #plot validation accuracy line
    plt.title(f'Model {stock}: Training and validation accuracy') #add title
    plt.legend() #add legend
    ax.spines['right'].set_visible(False) #hide right axis
    ax.spines['top'].set_visible(False) #hide top axis
    
    plt.figure() #plot figure
    plt.savefig(f"{stock}_accuracy.jpg")
    
    fig,ax = plt.subplots() #new figure
    plt.plot(epochs, loss, 'blue', label='Training loss') #plot training loss line
    plt.plot(epochs, val_loss, 'red', label='Validation loss') #plot validation loss line
    plt.title(f'Model {stock}: Training and validation loss') #add title
    plt.legend() #add legend
    ax.spines['right'].set_visible(False) #hide right axis
    ax.spines['top'].set_visible(False) #hide top axis
    
    plt.show() #show plot
    plt.savefig(f"{stock}_loss.jpg")
    
    
        #Create confusion matrix function
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title=f'Model {stock}: Training Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"{stock}_confusion_matrix.jpg")
    
    
    # Predict the values from the validation dataset
    pred_label = model.predict(TSG_test)
    # Convert predictions probabilities to predictions
    pred_label_classes = np.where(pred_label > 0.5, 1, 0) 
    # Convert validation observations to one hot vectors
    label_true = y_test[1:]
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(label_true, pred_label_classes) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(2)) 

#Export Model KPIs as CSV file
KPI.to_csv("Model_KPIs.csv",index=True)

