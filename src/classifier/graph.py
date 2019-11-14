from keras.layers import Input, Dense
from keras.models import Model
import keras as Keras
from keras.callbacks import ModelCheckpoint
import sys
import os
sys.path.append(os.path.abspath("../libs"))
from data import MNISTProcessor
import config as conf

#load data
data_processor = MNISTProcessor(conf.data_path, conf.train_labels, 
                                conf.train_images, '', '')
x_data_train, y_data_train = data_processor.load_train(normalize=True).get_training_data()

# Load the autoencoder model, including its weights and then process images
autoencoder = Keras.models.load_model(conf.autoencoder_model_path)
clean_images = autoencoder.predict(x_data_train)

#initialize the classification network
input_layer = Input(shape=(784,))
network = Dense(80, activation='tanh')(input_layer)
network = Dense(40, activation='tanh')(network)
network = Dense(20, activation='tanh')(network)
network = Dense(10, activation='tanh')(network)

classifier = Model(input_layer, network)
classifier.compile(optimizer='adadelta', loss='MSE', metrics=['accuracy'])

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=conf.checkpoint_path, save_weights_only=True, verbose=1)

#load an existing model to continue training
#classifier.load_weights(conf.checkpoint_path)

#run the model
classifier.fit(x_data_train, y_data_train,
                epochs=conf.epochs,
                batch_size=conf.batch_size,
                shuffle=True,
                callbacks=[cp_callback])

#save production version of the model
classifier.save(conf.final_model_path) 
classifier.summary()  