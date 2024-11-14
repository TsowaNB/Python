import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers, models
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

# SNIPPET 2
(t_train, t_test), t_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# SNIPPET 3
def prepproces(image, label):
    image = tf.cast(image, tf.float32)/255.0
    return image, label

# SNIPPET 4
t_train = t_train.map(prepproces, num_parallel_calls=tf.data.experimental.AUTOTUNE)
t_test = t_test.map(prepproces, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# SNIPPET 5
batch_size = 32
t_train = t_train.batch(batch_size)
t_test = t_test.batch(batch_size)

# SNIPPET 6
for image_batch, label_batch in t_train.take(1):
   print(f"image shape:{image_batch.shape}") 
   print(f"label batch:{label_batch.numpy()}")                    
                      



