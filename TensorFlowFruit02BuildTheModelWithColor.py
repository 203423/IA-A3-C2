import os
import cv2
import numpy as np
from numpy import load
from numpy.core.defchararray import index
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import tensorflow as tf 
from tensorflow import keras
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


class_names = ["apple","cucumber","greentomato","Guineo","lemon","lemontangerine","mango","potato","Uva"]
train_data = load('./temp/train_data.npy')
train_data_lables = load('./temp/train_data_labels.npy')
test_data = load('./temp/test_data.npy')
test_data_big = load('./temp/test_data_big.npy')
test_data_labels = load('./temp/test_data_labels.npy')

print("Finish loading the data ")





print("train shape : ", train_data.shape)
print("train lables shape : ", train_data_lables.shape)
print("test data shape:", test_data.shape)
print("test data labels shape:", test_data_labels.shape)


train_data = train_data / 255.0
test_data = test_data / 255.0



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28,3)),

    # lets define the hidden layer.
    # we dont know what is the exact number , so will try with 512 neurons
    keras.layers.Dense(100,activation='relu'), # relu has no negative values.

    keras.layers.Dense(9,activation='softmax') 
])

print('Finish build the model skeleton')

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print('Finish compile the model')

model.fit(train_data,train_data_lables,epochs=120)


test_loss , test_acc = model.evaluate(test_data,test_data_labels,verbose=1)
print("*******************         Test accuracy : ", test_acc)

camara = cv2.VideoCapture(0)

frame_weight = 700
frame_height = 700

predictions=model.predict(test_data)
model.save('FruitModel.h5')
print("Modelo guardado")



print ('The predicted class index :')



for predict , test_label in zip(predictions,test_data_labels):
    class_index = np.argmax(predict)
    class_name_predict = class_names[class_index]

    class_name_original = class_names[test_label]

    print('Predicted class :',class_name_predict , '     Original / real class name :', class_name_original )


def show_output(x):
    class_index = np.argmax(predictions[x])
    class_name = class_names[class_index]
    demoImage = test_data_big[x]
    cv2.putText(demoImage,class_name,(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
    cv2.imshow('demoImage',demoImage)
    cv2.waitKey(0)
def create_confusion_matrix(predictions):
    y_pred = np.argmax(predictions,axis=1).flatten()
    y_true = np.asarray(test_data_labels).flatten()
    print(("Test accuracy: {:.2f}%".format(test_acc * 100)))
    matc = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_mat=matc,figsize=(10,10),class_names=class_names,show_normed=False)
    plt.tight_layout()
    plt.show()




def salidas():
    show_output(np.random.randint(0,4))
    show_output(np.random.randint(5,9))
    show_output(np.random.randint(10,14))
    show_output(np.random.randint(15,19))
    show_output(np.random.randint(20,24))
    show_output(np.random.randint(25,29))
    show_output(np.random.randint(30,34))
    show_output(np.random.randint(35,39))
    show_output(np.random.randint(40,44))
# salidas()
time.sleep(2)
# create_confusion_matrix(predictions)






