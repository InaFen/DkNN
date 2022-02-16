from models import cifar10_cnn, cifar10_resnet50
from get_data_refactored import get_data
import tensorflow as tf
import os
import subprocess
import sys


NUM_DATA_POINTS = 10000 # how many of member/non member elements are used
PATH_MODEL = "/home/inafen/jupyter_notebooks/dataset_inference/model_resnet50"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

#get member data
members, test_data = get_data('cifar10', augmentation=False, batch_size=NUM_DATA_POINTS, indices_to_use=range(0, 25000))
iterator_members = members.next()  #TODO iterator_members good name choice?
member_data = iterator_members[0]
member_labels = iterator_members[1]

#load model or build and save model
try:
    model = tf.keras.models.load_model(PATH_MODEL)
except:
    model = cifar10_cnn()
    # compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics="accuracy",
    )
    # train the model
    # if you want specify batch size, learning rates etc.
    history = model.fit(member_data, member_labels, epochs=50, batch_size=64)
    # export model
    model.save(PATH_MODEL)
#train_accuracy = model.evaluate(member_data, member_labels)

#TODO different amount of samples, loops, see e-Mail

#generate features
subprocess.run(['which', 'python'])
subprocess.run(['/home/inafen/.conda/envs/mia2/bin/python', 'cleverhans-dataset-inference/src/generate_features.py'])

#notebook
