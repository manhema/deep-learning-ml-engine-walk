#!/bin/bash

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMNS = [
    'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'
]

CSV_COLUMN_DEFAULTS = [[0], [0], [0], [0], [0]]

LABEL_COLUMN = 'Species'

# LABELS = ['Setosa', 'Versicolor', 'Virginica']
LABELS = [0, 1, 2]
