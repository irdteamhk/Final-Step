# Final-Step

Congrat! You passed all challenges!

Please follow below steps to complete:
1. Fork this repo
2. Push your source code to the repo. (Source code only, don't push any data)
3. Create a pull request towards this repo
4. Copy the pull request id (e.g. #XXXXX)
5. Send it to where the bot told you

```
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot #original
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
from tensorflow import keras
```

```
def get_images(directory):
    Images = []
    Labels = [] 
    label = 0
    
    for labels in os.listdir(directory): 
        if labels == 'glacier': 
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3
        
        for image_file in os.listdir(directory+labels): 
            image = cv2.imread(directory+labels+r'/'+image_file) 
            image = cv2.resize(image,(150,150)) 
            Images.append(image)
            Labels.append(label)
    
    return shuffle(Images,Labels,random_state=817328462) 
    
    def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}
    
    return labels[class_code]
```

```
Images, Labels = get_images('C:/Users/dexte/OneDrive - The Chinese University of Hong Kong/FDM group/Manulife challenge/mil_ird_challenge/train/train/') 
Images = np.array(Images) 
Labels = np.array(Labels)
```

```
model = Models.Sequential()
model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.2))
model.add(Layers.Dense(6,activation='softmax'))
model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
```

```
SVG(model_to_dot(model).create(prog='dot', format='svg')) 
Utils.plot_model(model,to_file='model.png',show_shapes=True)  
```

```
trained = model.fit(Images,Labels,epochs=15,validation_split=0.30)
Train on 9627 samples, validate on 4126 samples

Epoch 1/15
9627/9627 [==============================] - 1084s 113ms/sample - loss: 1.4044 - acc: 0.4778 - val_loss: 1.0369 - val_acc: 0.5744
Epoch 2/15
9627/9627 [==============================] - 739s 77ms/sample - loss: 1.0104 - acc: 0.6175 - val_loss: 0.8402 - val_acc: 0.6970
Epoch 3/15
9627/9627 [==============================] - 832s 86ms/sample - loss: 0.8503 - acc: 0.6817 - val_loss: 0.7254 - val_acc: 0.7307
Epoch 4/15
9627/9627 [==============================] - 805s 84ms/sample - loss: 0.7503 - acc: 0.7246 - val_loss: 0.7020 - val_acc: 0.7443
Epoch 5/15
9627/9627 [==============================] - 746s 77ms/sample - loss: 0.6552 - acc: 0.7650 - val_loss: 0.6613 - val_acc: 0.7651
Epoch 6/15
9627/9627 [==============================] - 886s 92ms/sample - loss: 0.6124 - acc: 0.7807 - val_loss: 0.6177 - val_acc: 0.7770
Epoch 7/15
9627/9627 [==============================] - 781s 81ms/sample - loss: 0.5497 - acc: 0.8015 - val_loss: 0.6114 - val_acc: 0.7811
Epoch 8/15
9627/9627 [==============================] - 1382s 144ms/sample - loss: 0.5075 - acc: 0.8180 - val_loss: 0.6785 - val_acc: 0.7571
Epoch 9/15
9627/9627 [==============================] - 1116s 116ms/sample - loss: 0.4580 - acc: 0.8382 - val_loss: 0.6523 - val_acc: 0.7782
Epoch 10/15
9627/9627 [==============================] - 715s 74ms/sample - loss: 0.4232 - acc: 0.8513 - val_loss: 0.5744 - val_acc: 0.8017
Epoch 11/15
9627/9627 [==============================] - 656s 68ms/sample - loss: 0.3777 - acc: 0.8666 - val_loss: 0.5943 - val_acc: 0.8034
Epoch 12/15
9627/9627 [==============================] - 776s 81ms/sample - loss: 0.3441 - acc: 0.8781 - val_loss: 0.5781 - val_acc: 0.8008
Epoch 13/15
9627/9627 [==============================] - 832s 86ms/sample - loss: 0.2973 - acc: 0.8997 - val_loss: 0.6002 - val_acc: 0.8122
Epoch 14/15
9627/9627 [==============================] - 884s 92ms/sample - loss: 0.2863 - acc: 0.9017 - val_loss: 0.6408 - val_acc: 0.8017
Epoch 15/15
9627/9627 [==============================] - 810s 84ms/sample - loss: 0.2201 - acc: 0.9224 - val_loss: 0.6567 - val_acc: 0.8165
```

```
model.save('C:/Users/dexte/OneDrive - The Chinese University of Hong Kong/FDM group/Manulife challenge/mil_ird_challenge_others/manulifeModel')
```

```
test_images,test_labels = get_images('C:/Users/dexte/OneDrive - The Chinese University of Hong Kong/FDM group/Manulife challenge/mil_ird_challenge/test/test/')
test_images = np.array(test_images)
test_labels = np.array(test_labels)
model.evaluate(test_images,test_labels, verbose=1)
```

```
pred_images,no_labels = get_images('C:/Users/dexte/OneDrive - The Chinese University of Hong Kong/FDM group/Manulife challenge/mil_ird_challenge/pred/')
pred_images = np.array(pred_images)
pred_images.shape

fig = plot.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    pred_image = np.array([pred_images[i]])
    pred_class = get_classlabel(model.predict_classes(pred_image)[0])
    pred_prob = model.predict(pred_image).reshape(6)
    for j in range(2):
        if (j%2) == 0:
            ax = plot.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plot.Subplot(fig, inner[j])
            ax.bar([0,1,2,3,4,5],pred_prob)
            fig.add_subplot(ax)


fig.show()
```








    
    
