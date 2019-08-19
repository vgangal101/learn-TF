from __future__ import absolute_import, division, print_function,unicode_literals

# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def main():
  # get data
  fashion_mnist = keras.datasets.fashion_mnist

  # load data into arrays img,label
  (train_images,train_labels), (test_images,test_labels) = fashion_mnist.load_data()

  class_names= ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

  #  print(train_images.shape)
  #  print(train_labels)
  #  print(test_images.shape)
  #  print(len(test_labels))

 #  plt.figure()
 #  plt.imshow(train_images[0])
 #  plt.colorbar()
 #  plt.grid(False)
 #  plt.show()
 #

  # need to scale values to range of 0 to 1
  train_images = train_images / 255.0
  test_images = test_images / 255.0


  # build the model 

  model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                            keras.layers.Dense(128, activation = tf.nn.relu),
                            keras.layers.Dense(10,activation = tf.nn.softmax)])
  
  # compile with optimizer,loss fxn, and metric i.e accuracy
  model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

  # train the model (call the fit fxn)
  model.fit(train_images, train_labels, epochs=5)

  # run evaluation
  test_loss, test_acc = model.evaluate(test_images,test_labels)

  print('Test accuracy:',test_acc)
  


  # make predictions
  predictions = model.predict(test_images)
  
  # print prediction confidences 
  print(predictions[0])  
  
  # say which the model's best guess
  print(np.argmax(predictions[0]))



  # using trained model to make a prediction about a single image 
  img = test_images[0]
  

  # add the image to a batch where its the only member.
  img = (np.expand_dims(img,0))
  print(img.shape)
  
  # run predictions on this img
  single_prediction = model.predict(img)
  print(single_prediction)

  # the actual predicition 
  prediction_result = np.argmax(single_prediction[0])
  print(prediction_result)

if __name__ == '__main__':
    main()
