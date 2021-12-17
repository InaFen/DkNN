from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, Activation
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange

def make_basic_cnn():
  """Build a basic CNN.

  :return: CNN model
  """
  shape = (28, 28, 1)
  i = Input(shape=shape)
  x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(i)
  x = MaxPooling2D()(x)
  x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
  x = MaxPooling2D()(x)
  x = Flatten()(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(10)(x)
  # leave out the last Softmax in order to have logits and no softmax
  # x = Activation('softmax')(x)
  model = Model(i, x)
  return model


def make_shallow_basic_cnn():
  """Build a basic CNN which lacks complexity.
  The reason for this is to have a fast CNN for testing purposes.

  :return: CNN model
  """
  shape = (28, 28, 1)
  i = Input(shape=shape)
  x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(i)
  x = MaxPooling2D()(x)
  x = Flatten()(x)
  x = Dense(10)(x)
  model = Model(i, x)
  return model


def plot_reliability_diagram(confidence, labels, filepath):
  """
  Takes in confidence values for predictions and correct
  labels for the data, plots a reliability diagram.

  :param confidence: nb_samples x nb_classes (e.g., output of softmax)
  :param labels: vector of nb_samples
  :param filepath: where to save the diagram
  """
  assert len(confidence.shape) == 2
  assert len(labels.shape) == 1
  assert confidence.shape[0] == labels.shape[0]
  print('Saving reliability diagram at: ' + str(filepath))
  if confidence.max() <= 1.:
        # confidence array is output of softmax
    bins_start = [b / 10. for b in xrange(0, 10)]
    bins_end = [b / 10. for b in xrange(1, 11)]
    bins_center = [(b + .5) / 10. for b in xrange(0, 10)]
    preds_conf = np.max(confidence, axis=1)
    preds_l = np.argmax(confidence, axis=1)
  else:
    raise ValueError('Confidence values go above 1.')

  print(preds_conf.shape, preds_l.shape)

  # Create var for reliability diagram
  # Will contain mean accuracies for each bin
  reliability_diag = []
  num_points = []  # keeps the number of points in each bar

  # Find average accuracy per confidence bin
  for bin_start, bin_end in zip(bins_start, bins_end):
    above = preds_conf >= bin_start
    if bin_end == 1.:
      below = preds_conf <= bin_end
    else:
      below = preds_conf < bin_end
    mask = np.multiply(above, below)
    num_points.append(np.sum(mask))
    bin_mean_acc = max(0, np.mean(preds_l[mask] == labels[mask]))
    reliability_diag.append(bin_mean_acc)

  # Plot diagram
  assert len(reliability_diag) == len(bins_center)
  print(reliability_diag)
  print(bins_center)
  print(num_points)
  fig, ax1 = plt.subplots()
  _ = ax1.bar(bins_center, reliability_diag, width=.1, alpha=0.8)
  plt.xlim([0, 1.])
  ax1.set_ylim([0, 1.])

  ax2 = ax1.twinx()
  print(sum(num_points))
  ax2.plot(bins_center, num_points, color='r', linestyle='-', linewidth=7.0)
  ax2.set_ylabel('Number of points in the data', fontsize=16, color='r')

  if len(np.argwhere(confidence[0] != 0.)) == 1:
    # This is a DkNN diagram
    ax1.set_xlabel('Prediction Credibility', fontsize=16)
  else:
    # This is a softmax diagram
    ax1.set_xlabel('Prediction Confidence', fontsize=16)
  ax1.set_ylabel('Prediction Accuracy', fontsize=16)
  ax1.tick_params(axis='both', labelsize=14)
  ax2.tick_params(axis='both', labelsize=14, colors='r')
  fig.tight_layout()
  plt.savefig(filepath, bbox_inches='tight')
