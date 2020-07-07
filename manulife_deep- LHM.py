# Important! This is a combine result from Neural Network and Random forest
# The criteria is that the result will follow Neural Network. The first priority of Neural Network predicting label 3,
# and second priority of random forest predicting label 4. And the rest would follow Neural Network
# Random forest script is commented
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import os


#cifar = tf.keras.datasets.cifar10

#(x_train, y_train), (x_test, y_test) = cifar.load_data()
def load_image(path):
    image = []
    path = path
    i=0
    for filename in os.listdir(path):
        if i>2000:
            break
        else:
            i+=1
        if filename.split(".")[-1] == 'jpg':

            imgs = img.imread(os.path.join(path, filename))
            imgs = np.asarray(imgs)
            if imgs.shape[0] != 150:
                continue
            elif imgs.shape[1] != 150:
                continue
            image.append(imgs)




    return image


def load_image1(path):
    image = []
    path = path
    filenameList = []
    i=0
    for filename in os.listdir(path):
        if i>2100:
            break
        else:
            i+=1
        if filename.split(".")[-1] == 'jpg':

            imgs = img.imread(os.path.join(path, filename))
            plt.title(filename)
            filenameList.append(filename)
            #plt.imshow(imgs)
            #plt.show()
            image.append(imgs)




    return image,filenameList

train_0 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/0")
train_0= np.asarray(train_0)

num_of_train_0 = len(train_0)
train_1 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/1")
train_1 = np.asarray(train_1)
num_of_train_1 = len(train_1)


train_2 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/2")
num_of_train_2 = len(train_2)
train_2 = np.asarray(train_2)


train_3 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/3")
num_of_train_3 = len(train_3)
train_3 = np.asarray(train_3)

train_4 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/4")
num_of_train_4 = len(train_4)
train_4 = np.asarray(train_4)

train_5 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/5")
num_of_train_5 = len(train_5)
train_5 = np.asarray(train_5)


test_0 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/0")
num_of_test_0 = len(test_0)
test_0 = np.asarray(test_0)

test_1 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/1")
num_of_test_1 = len(test_1)
test_1 = np.asarray(test_1)

test_2 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/2")
num_of_test_2 = len(test_2)
test_2 = np.asarray(test_2)

test_3 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/3")
num_of_test_3 = len(test_3)
test_3 = np.asarray(test_3)

test_4 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/4")
num_of_test_4 = len(test_4)
test_4 = np.asarray(test_4)

test_5 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/5")
num_of_test_5 = len(test_5)
test_5 = np.asarray(test_5)

x_train = np.concatenate((train_0 , train_1,train_2,train_3,train_4,train_5), axis=0)
x_test = np.concatenate((test_0 , test_1,test_2,test_3,test_4,test_5), axis=0)
y_label_train_0 = np.zeros(num_of_train_0)
y_label_train_1 = np.full(num_of_train_1,1)
y_label_train_2 = np.full(num_of_train_2,2)
y_label_train_3 = np.full(num_of_train_3,3)
y_label_train_4 = np.full(num_of_train_4,4)
y_label_train_5 = np.full(num_of_train_5,5)

y_label_test_0 = np.zeros(num_of_test_0)
y_label_test_1 = np.full(num_of_test_1,1)
y_label_test_2 = np.full(num_of_test_2,2)
y_label_test_3 = np.full(num_of_test_3,3)
y_label_test_4 = np.full(num_of_test_4,4)
y_label_test_5 = np.full(num_of_test_5,5)
y_train = np.concatenate((y_label_train_0,y_label_train_1,y_label_train_2,y_label_train_3,y_label_train_4,y_label_train_5))
y_test = np.concatenate((y_label_test_0,y_label_test_1,y_label_test_2,y_label_test_3,y_label_test_4,y_label_test_5))

x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
print(y_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, input_dim=5,
                          kernel_regularizer=tf.keras.regularizers.l1(0.03),
                          activity_regularizer=tf.keras.regularizers.l2(0.03)),
    tf.keras.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=40, epochs=11)

model.evaluate(x_test,  y_test, verbose=2)

target1,filenameList = load_image1("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/pred")
target1 = np.asarray(target1)

for i in range(98):
    prediction = np.argmax(model.predict(np.expand_dims(target1[i],axis=0)))
    #prediction = model.predict_classes([target1[i])
    #prediction = model.predict([pred_generator[i]])
    print(filenameList[i])
    print(prediction)


## Random Forest Script

'''
import numpy as np
from skimage.feature import hog
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm
import skimage
def load_image(path):
    image = []
    path = path
    i=0
    for filename in os.listdir(path):
        if i>2000:
            break
        else:
            i+=1
        if filename.split(".")[-1] == 'jpg':

            imgs = img.imread(os.path.join(path, filename))
            imgs = np.asarray(imgs)
            if imgs.shape[0] != 150:
                continue
            elif imgs.shape[1] != 150:
                continue
            image.append(imgs)




    return image


def load_image1(path):
    image = []
    path = path
    filenameList = []
    i=0
    for filename in os.listdir(path):
        if i>200:
            break
        else:
            i+=1
        if filename.split(".")[-1] == 'jpg':

            imgs = img.imread(os.path.join(path, filename))
            plt.title(filename)
            filenameList.append(filename)
            #plt.imshow(imgs)
            #plt.show()
            image.append(imgs)




    return image,filenameList
def processing(image):
    train_prepared, hog_image = hog(image, pixels_per_cell=(24, 24), cells_per_block=(2, 2), orientations=8,
                                      visualize=True, block_norm='L2-Hys')


    #print('number of pixels: ', image.shape[0] * image.shape[1])
    #print('number of hog features: ', train_prepared.shape[0])
    #print('number of hog features: ', train_prepared.shape)
    return train_prepared

def transform(dataset):
    number = len(dataset)
    print(len(dataset))
    k =0
    train_prepared = np.zeros((number,22500), dtype='float32')
    for i in range(number):
        temp = skimage.color.rgb2gray(dataset[i])
        temp = img.thumbnail((64,64),Im)
        temp = temp.flatten()
        print(k)
        k += 1
        for j in range(22500):
            train_prepared[i][j] = temp[j]



    return train_prepared

train_0 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/0")
num_of_train_0 = len(train_0)
train_prepared0 =(transform(train_0))
#print(train_prepared0.shape)

train_1 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/1")
num_of_train_1 = len(train_1)
train_prepared1= transform(train_1)

train_2 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/2")
num_of_train_2 = len(train_2)
train_prepared2 =transform(train_2)

train_3 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/3")
num_of_train_3 = len(train_3)
train_prepared3= transform(train_3)

train_4 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/4")
num_of_train_4 = len(train_4)
train_prepared4 =transform(train_4)

train_5 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/train/5")
num_of_train_5 = len(train_5)
train_prepared5= transform(train_5)

train_prepared = np.concatenate((train_prepared0 , train_prepared1,train_prepared2,train_prepared3,train_prepared4,train_prepared5), axis=0)
pca = PCA(n_components=100)
pca.fit(train_prepared)
#train_prepared = pca.transform(train_prepared)
#print(pca.explained_variance_)
#train_prepared = pca.transform(train_prepared)
#print(train_prepared.shape)


test_0 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/0")
num_of_test_0 = len(test_0)
test_prepared0 = transform(test_0)
print(test_prepared0.shape)

test_1 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/1")
num_of_test_1 = len(test_1)
test_prepared1 = transform(test_1)

test_2 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/2")
num_of_test_2 = len(test_2)
test_prepared2 = transform(test_2)

test_3 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/3")
num_of_test_3 = len(test_3)
test_prepared3 = transform(test_3)

test_4 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/4")
num_of_test_4 = len(test_4)
test_prepared4 = transform(test_4)

test_5 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/5")
num_of_test_5 = len(test_5)
test_prepared5 = transform(test_5)

test_prepared = np.concatenate((test_prepared0 , test_prepared1,test_prepared2,test_prepared3,test_prepared4,test_prepared5), axis=0)


pca = PCA(n_components=100)
pca.fit(test_prepared)
print(pca.explained_variance_)
test_prepared = pca.transform(test_prepared)
print(test_prepared.shape)

target,filenameList = load_image1("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/pred")
num_of_target = len(target)
target_prepared = transform(target)
pca = PCA(n_components=100)
pca.fit(target_prepared)
#target_prepared = pca.transform(target_prepared)


test_prepared0 = transform(test_0)
print(test_prepared0.shape)
test_prepared1 = transform(test_1)

test_prepared2 = transform(test_2)

test_prepared3 = transform(test_3)

test_prepared4 = transform(test_4)

test_prepared5 = transform(test_5)

test_prepared = np.concatenate((test_prepared0 , test_prepared1,test_prepared2,test_prepared3,test_prepared4,test_prepared5), axis=0)
pca = PCA(n_components=100)
pca.fit(test_prepared)
print(pca.explained_variance_)
test_prepared = pca.transform(test_prepared)
print(test_prepared.shape)

train_prepared0 =transform(train_0)
train_prepared1= transform(train_1)
train_prepared2 =transform(train_2)
train_prepared3= transform(train_3)
train_prepared4 =transform(train_4)
train_prepared5= transform(train_5)
train_prepared = np.concatenate((train_prepared0 , train_prepared1,train_prepared2,train_prepared3,train_prepared4,train_prepared5), axis=0)
pca = PCA(n_components=100)
pca.fit(train_prepared)
train_prepared = pca.transform(train_prepared)

#num_of_train = len(train_prepared)
#num_of_test = len(test_prepared)


y_label_train_0 = np.zeros(num_of_train_0)
y_label_train_1 = np.full(num_of_train_1,1)
y_label_train_2 = np.full(num_of_train_2,2)
y_label_train_3 = np.full(num_of_train_3,3)
y_label_train_4 = np.full(num_of_train_4,4)
y_label_train_5 = np.full(num_of_train_5,5)

#y_label_test_0 = np.zeros(num_of_test_0)
#y_label_test_1 = np.full(num_of_test_1,1)
#y_label_test_2 = np.full(num_of_test_2,2)
#y_label_test_3 = np.full(num_of_test_3,3)
#y_label_test_4 = np.full(num_of_test_4,4)
#y_label_test_5 = np.full(num_of_test_5,5)

y_label_train = np.concatenate((y_label_train_0,y_label_train_1,y_label_train_2,y_label_train_3,y_label_train_4,y_label_train_5))
#y_label_test = np.concatenate((y_label_test_0,y_label_test_1,y_label_test_2,y_label_test_3,y_label_test_4,y_label_test_5))
#test_0 = load_image("/Users/homan/Desktop/FDM/Manulife/mil_ird_challenge/mil_ird_challenge/test/0")
#train_0_label = np.zeros(len(train_0))
clf = RandomForestClassifier(n_estimators=120)



clf.fit(train_prepared,y_label_train)
preds = clf.predict(target_prepared)
print(filenameList)
print(preds)
#print("Accuracy:", accuracy_score(y_label_test,preds))
'''