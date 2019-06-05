from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import trainer.featurizer as featurizer


def build_estimator(config, hidden_units=None):
    classifier = tf.estimator.DNNClassifier(
        config=config,
        feature_columns=featurizer.INPUT_COLUMNS,
        hidden_units=hidden_units or [10, 10],
        n_classes=3,
    )
    return classifier
