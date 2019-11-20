from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import flags
from tensorflow.keras import optimizers
import sys, os
import config as conf

#set up and parse custom flags
flags.DEFINE_integer('model_version', conf.version, "Width of the image")
flags.DEFINE_boolean('rebuild', False, "Drop the checkpoint weights and rebuild model from scratch")
flags.DEFINE_string('lib_folder', conf.lib_folder, "Local library folder")
FLAGS = flags.FLAGS

#mount the library folder
sys.path.append(os.path.abspath(FLAGS.lib_folder))
from data import MNISTProcessor
import visualizer as v

#load data
data_processor = MNISTProcessor(conf.data_path, conf.train_labels, 
                                conf.train_images, '', '')                               
x_data_train, y_data_train = data_processor.load_train(normalize=True).get_training_data()

#initialize the network
input_layer = Input(shape=(784,), name='input')
network = Dense(152, activation='tanh', name='dense_1')(input_layer)
network = Dense(76, activation='tanh', name='dense_2')(network)
network = Dense(38, activation='tanh', name='dense_3')(network)
network = Dense(4, activation='tanh', name='dense_4')(network)
network = Dense(38, activation='tanh', name='dense_5')(network)
network = Dense(76, activation='tanh', name='dense_6')(network)
network = Dense(152, activation='tanh', name='dense_7')(network)
output = Dense(784, activation='tanh', name='output')(network)

autoencoder = Model(inputs=input_layer, outputs=output, name='autoencoder')
autoencoder.compile(optimizer=optimizers.Adadelta(learning_rate=1.0), loss='MSE', metrics=['accuracy'])

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=conf.checkpoint_path, save_weights_only=True, verbose=1)

#load an existing model to continue training
if(not FLAGS.rebuild):
    try:
        autoencoder.load_weights(conf.checkpoint_path)
    except:
        print('No checkpoint found, building filters from scratch.')

#run the training
autoencoder.fit(x_data_train, x_data_train,
                epochs=conf.epochs,
                batch_size=conf.batch_size,
                shuffle=True,
                callbacks=[cp_callback])

#save the production version of the model
try:
    os.mkdir(conf.final_model_path + '/' + str(FLAGS.model_version))
except OSError:
    print ("Creation of the directory %s failed" % (conf.final_model_path + '/' + str(FLAGS.model_version)))
  
autoencoder.save(conf.final_model_path + '/' + str(FLAGS.model_version), overwrite=True, save_format='tf') 

autoencoder.summary()   

# run a sample for visualization
clean_images = autoencoder.predict(x_data_train)
v.visualize_autoencoding(x_data_train, clean_images, digits_to_show=10)