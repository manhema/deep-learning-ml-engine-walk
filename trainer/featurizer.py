import tensorflow as tf

# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [
    # Continuous base columns.
    tf.feature_column.numeric_column('SepalLength'),
    tf.feature_column.numeric_column('SepalWidth'),
    tf.feature_column.numeric_column('PetalLength'),
    tf.feature_column.numeric_column('PetalWidth'),
    tf.feature_column.numeric_column('Species'),

    # Categorical base columns

    # For categorical columns with known values we can provide lists
    # of values ahead of time.
    # tf.feature_column.categorical_column_with_vocabulary_list(
    #     'Species', ['Setosa', 'Versicolor', 'Virginica']),
]


def get_dnn_columns():
    return INPUT_COLUMNS[:-1]
