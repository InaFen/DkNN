"""
Code source: https://github.com/fraboeni/membership-risk/blob/master/code_base/model_training.py
"""

import os
import tensorflow as tf

from utils.utils_data.get_data import get_data
from cifar10_models import MODELS
from tensorflow.keras.models import Model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#TODO epochs to 50
def main(dataset='cifar10', model_name='cifar10_cnn', augment=True, batch_size=64, lr=0.001, optim="Adam",
         momentum=0.9, nesterov=False, epochs=1, early_stop=True, save_model=True, log_training=True,
         logdir='log_dir/models/', from_logits=True):
    train_data, test_data = get_data(dataset, augmentation=augment, batch_size=batch_size,
                                     indices_to_use=range(0, 25000))
    if optim == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr,
                                            momentum=momentum,
                                            nesterov=nesterov)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if from_logits:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model = MODELS[model_name](from_logits=from_logits )
    model.build_graph().summary() #TODO
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy']
                  )
    model_id = 1004  # Todo parse from config file

    callbacks = []

    if early_stop:
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        callbacks.append(early_stop_callback)

    if log_training:
        # TODO change logfile back
        logfile = "/home/inafen/jupyter_notebooks/1004_model_cifar10_temp.csv"
        #logfile = os.getcwd() + '/../' + logdir + dataset + '/' + str(model_id) + '_' + model_name + '.csv'
        print(logfile)
        logging_callback = tf.keras.callbacks.CSVLogger(logfile, separator=",", append=False)
        callbacks.append(logging_callback)

    history = model.fit(train_data,
                        validation_data=test_data,
                        epochs=epochs,
                        callbacks=callbacks,
                        )
    print(history.history)
    """
    # summarize filter shapes per layer
    # we are only interested in convolutional layers
    # print("Layer names (and shapes) of the model:")
    layer_indices = []
    nb_layers = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if ("conv" not in layer.name) and ("dense" not in layer.name):
            # print(layer.name)
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        # print(layer.name, filters.shape)
        nb_layers.append(layer.name)
        layer_indices.append(i)
    print(nb_layers)
    print(nb_layers)
    outputs = [model.get_layer(name=layer).output for layer in nb_layers[1:]]
    x = tf.keras.layers.Input(shape=(32,32,3))
    model_mult = Model(inputs=[x], outputs=model.call(x))
    print(outputs)
    """
    if save_model:
        #TODO change location back
        model.save("/home/inafen/jupyter_notebooks/1004_model_cifar10_temp", save_format='tf')
        #model.save(os.getcwd() + '/../' + logdir + dataset + '/' + str(model_id) + '_' + model_name, save_format='tf')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', choices=['cifar10', 'fmnist'])
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--early_stop', type=bool, default=True)
    # parser.add_argument('--logdir', default=None)
    # args = parser.parse_args()
    main()  # **vars(args))
