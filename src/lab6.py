import csv
import itertools
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras.preprocessing.text
from matplotlib.axes import Axes
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dropout
from tqdm import tqdm
from tqdm.keras import TqdmCallback

# TODO: Update paths to match your directory structure
# fig_path = '/remote_home/EENG645a-Sp23/Lab6/Lab6-template/figures'
# model_path = '/remote_home/EENG645a-Sp23/Lab6/Lab6-template/models'
# log_path = '/remote_home/EENG645a-Sp23/Lab6/Lab6-template/logs'
# vocab_path = ...

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{fig_path}/{title}')

# TODO: Create methods for writing and reading vocabulary, and getting the dataset from the vocabulary
def write_vocabulary(vocabulary_filename: Path, vocabulary: [str]):
    """
    I reccomend looking at csv.writer()
    """
    pass

def read_vocabulary(vocabulary_filename: Path) -> [str]:
    """ 
    I reccomend looking at csv.reader()
    """
    vocabulary = []
    return vocabulary


def get_dataset(data_dir: Path,
                vocabulary_filename: Path,
                max_words: int,
                max_token_length: int,
                batch_size: int,
                validation_split: float,
                ) -> [tf.data.Dataset, tf.data.Dataset, [str]]:
    # make dataset objects using text_dataset_from_directory
    # unlike other training, recommend setting a seed
    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset

    # check our categories and number of classes
    categories = train_dataset.class_names
    num_classes = len(categories)

    # process the raw text into tokens. remember to remove punctuation
    # you may want to load your saved vocabulary into the layer here (if a saved vocabulary exists)
    vectorize_layer: TextVectorization

    # determine and save vocabulary for later use
    if not os.path.exists(vocabulary_filename):
        # get vocabulary
        write_vocabulary(vocabulary_filename=vocabulary_filename, vocabulary=vectorize_layer.get_vocabulary())
    else:
        # if we didn't already load the vocabulary to the vectorize_layer do it now
        print("reading in saved vocabulary")
        vocabulary = read_vocabulary(vocabulary_filename=vocabulary_filename)
        vectorize_layer.set_vocabulary(vocab=np.asarray(vocabulary))

    # do any other work to process dataset.
    # For example, you could apply your vectorize layer here on the text input stream
    # Also  you could process the output classes to be one hots if necessary
    # Remember at this point you data should be shuffled, batched, scaled, repeated, prefetched, cached or
    # anything else you want in your pipeline

    # print out a sample to make sure it looks alright. You can comment this out after confirming. 
    sample = next(train_ds.as_numpy_iterator())
    print(f"Input shape: {sample[0].shape}, Output shape: {sample[1].shape}")
    print(f"First sample raw input:\n{sample[0][0]}")
    # if you used an integer encoding on the text we can decode it to get the sentence with our vocabulary
    vocabulary_list: [str] = vectorize_layer.get_vocabulary()
    text_list: [str] = []
    for word_index in sample[0][0]:
        text_list.append(vocabulary_list[word_index])
    print(f"First sample input decoded:\n{' '.join(text_list)}")
    print(f"First sample label: {sample[1][0]}")

    return train_dataset, validation_dataset, categories


def sample_from_dataset(dataset: tf.data.Dataset,
                        eval_samples=6000):
    """
    Uses a model and a dataset to visualize the current predictions

    :param dataset: the dataset to use for generating predictions and comparing against truth
    :param eval_samples: the number of samples used to evaluate from the dataset
    """
    x_visualize = []
    y_visualize = []

    total_samples = 0
    img_batch: np.ndarray
    label_batch: np.ndarray
    for img_batch, label_batch in dataset.as_numpy_iterator():
        total_samples += img_batch.shape[0]
        x_visualize.append(img_batch)
        y_visualize.append(label_batch[:, None])
        if total_samples > eval_samples:
            break

    x_visualize = np.vstack(x_visualize)
    y_visualize = np.vstack(y_visualize)

    x_visualize = np.squeeze(x_visualize)
    y_visualize = np.squeeze(y_visualize)

    return x_visualize, y_visualize


def main():
    base_data_dir = Path('/opt', 'data', f'website_txt_splits_small')
    train_data_dir = base_data_dir / 'train'
    test_data_dir = base_data_dir / 'test'

    # If True, then retrain the model
    force_fit_model: bool

    validation_split: float

    vocabulary_filename = Path(vocab_path, 'vocabulary.tsv')
    model_name = f'{model_path}/model.h5'

    batch_size: int

    # format the input sequences
    max_token_length: int
    max_words: int

    train_ds, valid_ds, categories = get_dataset(data_dir=train_data_dir,
                                                 vocabulary_filename=vocabulary_filename,
                                                 max_words=max_words,
                                                 max_token_length=max_token_length,
                                                 batch_size=batch_size,
                                                 validation_split=validation_split)

    if force_fit_model or not os.path.exists(model_name):

        # make model using the dataset
        model: Model

        # save the model
        model.save(model_name)

    else:
        # load the saved model
        model = load_model(model_name)

    # get some samples from our dataset
    x_test, y_test = sample_from_dataset(dataset=train_ds,
                                         eval_samples=6000)

    # do prediction
    y_true: np.ndarray
    y_pred: np.ndarray

    # print the visualization metrics
    class_names = [cat[:15] for cat in categories]

    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred,
                                                        y_true=y_true)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    # plt.show()


if __name__ == "__main__":
    main()
