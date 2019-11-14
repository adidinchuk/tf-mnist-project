from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import sys
import os
sys.path.append(os.path.abspath("../libs"))
from data import MNISTProcessor
import visualizer as v
import config as conf

data_processor = MNISTProcessor(conf.data_path, conf.train_labels, 
                                conf.train_images, '', '')
                                
x_data_train, y_data_train = data_processor.load_train(normalize=True).get_training_data()

#initialize the network
input_layer = Input(shape=(784,))
network = Dense(152, activation='tanh')(input_layer)
network = Dense(76, activation='tanh')(network)
network = Dense(38, activation='tanh')(network)
network = Dense(4, activation='tanh')(network)
network = Dense(38, activation='tanh')(network)
network = Dense(76, activation='tanh')(network)
network = Dense(152, activation='tanh')(network)
network = Dense(784, activation='tanh')(network)

autoencoder = Model(input_layer, network)
autoencoder.compile(optimizer='adadelta', loss='MSE')

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=conf.checkpoint_path, save_weights_only=True, verbose=1)

#load an existing model to continue training
autoencoder.load_weights(conf.checkpoint_path)

autoencoder.fit(x_data_train, x_data_train,
                epochs=conf.epochs,
                batch_size=conf.batch_size,
                shuffle=True,
                callbacks=[cp_callback])

autoencoder.save(conf.final_model_path) 

autoencoder.summary()   

clean_images = autoencoder.predict(x_data_train)

v.visualize_autoencoding(x_data_train, clean_images, digits_to_show=10)