"""Train a baseline classifier on data/real and data/spoof.
WARNING: Split by subject/video/source, not by frame, for valid final evaluation.
This script trains only; the validation split is not a trustworthy independent test.
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
if not any((DATA/'real').glob('*')) or not any((DATA/'spoof').glob('*')):
    raise SystemExit('Add labeled image samples to data/real and data/spoof first.')
train = tf.keras.utils.image_dataset_from_directory(
    DATA, class_names=['real','spoof'], validation_split=.2, subset='training', seed=42,
    image_size=(160,160), batch_size=16)
valid = tf.keras.utils.image_dataset_from_directory(
    DATA, class_names=['real','spoof'], validation_split=.2, subset='validation', seed=42,
    image_size=(160,160), batch_size=16)
base = tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False,
                                         weights='imagenet', pooling='avg')
base.trainable = False
inputs = tf.keras.Input((160,160,3))
x = tf.keras.layers.RandomFlip('horizontal')(inputs)
x = tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)(x)
x = base(x, training=False)
x = tf.keras.layers.Dropout(.25)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
model.fit(train, validation_data=valid, epochs=8)
(ROOT/'models').mkdir(exist_ok=True)
model.save(ROOT/'models'/'anti_spoof.keras')
print('Saved model. Evaluate on a separate subject/video-disjoint test set before deployment.')
