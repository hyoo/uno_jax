import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")


def load(file_prefix, batch_size, cell_input_size=942, drug_input_size=5270):
    def _parse_record_fn(record):
        feature_map = {
            'AUC': tf.io.FixedLenFeature([1], tf.float32),
            'cell': tf.io.FixedLenFeature([cell_input_size], tf.float32),
            'drug': tf.io.FixedLenFeature([drug_input_size], tf.float32),
        }
        record_features = tf.io.parse_single_example(record, feature_map)
        cell = record_features['cell']
        drug = record_features['drug']
        label = record_features['AUC']
        return (cell, drug), label

    def _generate_dataset_fn(file_prefix, partition, batch_size):
        ds = tf.data.Dataset.list_files(
            f'{file_prefix}.{partition}.*.tfr'
        ).interleave(
            lambda x: tf.data.TFRecordDataset(x).map(_parse_record_fn),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if partition == 'train':
            ds = ds.repeat()

        ds = ds.batch(
            batch_size=batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _generate_dataset_fn(file_prefix, 'train', batch_size)
    val_ds = _generate_dataset_fn(file_prefix, 'val', batch_size)
    # test_ds = _generate_dataset_fn(file_prefix, 'test', batch_size)

    return train_ds, val_ds
