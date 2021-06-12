
import os
import re
import csv
import h5py
import glob
import shutil
import random
import numpy as np
import tensorflow as tf


num_classes = [2, 1] # grapy, edges

def extract_edges(mask):
    edges = tf.image.sobel_edges(mask)

    boundary = tf.zeros([256, 192, 1])
    for i in range(edges.shape[0]):
        region_bdry = edges[i]
        img = tf.reduce_sum(region_bdry ** 2, axis=-1) + 1e-12
        img = tf.math.sqrt(img)
        boundary += tf.cast(img>0.05, tf.float32)

    boundary = tf.cast(boundary>0, tf.float32)

    return boundary

def filterMask(mask, valList):
    newmask = tf.cast(mask == valList[0], tf.float32)
    for val in valList[1:]:
        newmask += tf.cast(mask == val, tf.float32)
    return newmask

def _filter(data, model_inputs):
    # string = data["category"]
    # return tf.strings.regex_full_match(string, "real-background")
    return True

def _parse(proto):
    keys_to_features = {
        "personno": tf.io.FixedLenFeature([], tf.int64),
        "person": tf.io.FixedLenFeature([], tf.string),
        "personMask": tf.io.FixedLenFeature([], tf.string),
        "densepose": tf.io.FixedLenFeature([], tf.string),
        "is_shorts": tf.io.FixedLenFeature([], tf.string),
        "is_short_sleeves": tf.io.FixedLenFeature([], tf.string),
        "category": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    image = (tf.cast(tf.image.decode_jpeg(parsed_features["person"], channels=3), tf.float32))

    """
    GRAPY
    New labels
        {
            0: Background+UpperClothes+Dresses, 
            1: Head, 
            2: Gloves+Accessory+Tie+Bag+Belt+Necklace, 
            3: Lower Body, 
            4: TorsoSkin,
            5: LeftArm, 
            6: RightArm,
        }
    """
    orig_grapy = tf.cast(tf.image.decode_png(parsed_features["personMask"], channels=1), tf.float32)
    new_mapping = tf.constant([0, 1, 1, 2, 1, 0, 0, 0, 3, 3, 4, 1, 3, 1, 5, 6, 3, 3, 3, 3, 0, 2, 2, 2, 2, 2])
    grapy = tf.cast(tf.gather(new_mapping, tf.cast(orig_grapy, dtype=tf.int32)), tf.float32)

    # Converting to silhouette
    full_body_grapy = tf.cast(grapy>0, tf.float32)

    half_body_mapping = tf.constant([0, 1, 1, 2, 1, 1, 1])
    half_body_grapy = tf.cast(tf.gather(half_body_mapping, tf.cast(grapy, dtype=tf.int32)), tf.float32)

    # Old grapy mapping to remove images with dresses from training set
    old_mapping = tf.constant([0, 1, 1, 8, 1, 2, 4, 2, 3, 3, 5, 1, 3, 1, 6, 7, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2])
    old_grapy = tf.cast(tf.gather(old_mapping, tf.cast(orig_grapy, dtype=tf.int32)), tf.float32)

    """
    DENSEPOSE
    Changes
        L, R Foot(4, 5) + Upper Leg (6, 7) + Lower Leg (8, 9) -> Legs (4)
        Re-label upper limbs
    
    New Labels
        0      = Background
        1      = Torso
        2      = Right Hand
        3      = Left Hand
        4      = Legs
        5 = Upper Arm Left
        6 = Upper Arm Right
        7 = Lower Arm Left
        8 = Lower Arm Right
        9 = Head
    """
    densepose = tf.cast(tf.image.decode_png(parsed_features["densepose"], channels=1), tf.float32)
    mapping = tf.constant([0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9])
    densepose = tf.gather(mapping, tf.cast(densepose, dtype=tf.int32))
    densepose = tf.cast(densepose, tf.float32)

    dp_seg = tf.cast(densepose[..., 0], tf.int32)
    dp_seg = tf.cast(tf.one_hot(dp_seg, depth=10), tf.float32)

    person_no = parsed_features["personno"]
    category = parsed_features["category"]
    
    if True:
        if tf.random.uniform([]) < 0.25:
            if tf.random.uniform([]) < 0.2:
                image = tf.image.random_contrast(image, 0.4, 0.6)
            if tf.random.uniform([]) > 0.5:
                image = tf.image.random_saturation(image, 0.5, 1)

    image = image / 127.5 - 1

    edges = extract_edges(tf.expand_dims(grapy, axis=0))

    grapy_one_hot = tf.cast(grapy[..., 0], tf.int32)
    grapy_one_hot = tf.cast(tf.one_hot(grapy_one_hot, depth=num_classes[0]), tf.float32)

    person_no = parsed_features["personno"]

    # if tf.strings.regex_full_match(category, "none"):
    #     if tf.random.uniform([]) < 0.05:
    #         category = tf.constant("real-background")

    # if tf.strings.regex_full_match(category, "real-background"):
    #     # Removing Grapy Dresses while training
    #     dresses = tf.cast(filterMask(old_grapy, [4]), dtype=tf.float32)
    #     if tf.reduce_sum(dresses) > 10:
    #         category = tf.constant("none")
        
    # TPU specific operation
    grapy = tf.reshape(grapy, (256, 192, 1))
    grapy = tf.cast(grapy, tf.float32)

    edges = tf.reshape(edges, (256, 192, 1))
    edges = tf.cast(edges, tf.float32)

    image = tf.reshape(image, (256, 192, 3))
    image = tf.cast(image, tf.float32)

    densepose = tf.reshape(densepose, (256, 192, 10))
    densepose = tf.cast(densepose, tf.float32)

    grapy_one_hot = tf.reshape(grapy_one_hot, (256, 192, num_classes[0]))
    grapy_one_hot = tf.cast(grapy_one_hot, tf.float32)

    # To filter dataset
    filterval = tf.zeros(shape=(256, 192, 1), dtype=tf.float32)
    if tf.strings.regex_full_match(category, "real-background"):
        filterval = tf.ones(shape=(256, 192, 1), dtype=tf.float32)
    
    model_inputs = image
    data = {
        'segmentations': grapy,
        'segmentations_half': half_body_grapy,
        'segmentations_full': full_body_grapy
    }
    
    return model_inputs, data


def create_dataset(parse_func, filter_func, tfrecord_path, num_data, batch_size, mode, data_split, device):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    if mode == "train":
        dataset = dataset.take(int(data_split * num_data))
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

    elif mode == "val":
        dataset = dataset.skip(int(data_split * num_data))
        dataset = dataset.take(int((1-data_split)*num_data))

    elif mode == "k_worst":
        dataset = dataset.take(data_split * num_data)

    dataset = dataset.map(
        parse_func,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.filter(filter_func)

    if mode != "k_worst":
        # num_lines = sum(1 for _ in dataset)
        num_lines = 15000
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        num_lines = num_data  # doesn't get used anywhere
        dataset = dataset.batch(batch_size, drop_remainder=False)

    if device != "colab_tpu":
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, num_lines
    else:
        return dataset

def define_dataset(tfrecord_path, batch_size, train=True, test=False):
    per_replica_train_batch_size = batch_size
    per_replica_val_batch_size = batch_size
    if test:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=5000,
            batch_size=per_replica_train_batch_size,
            mode="k_worst",
            data_split=1,
            device='gpu',
        )
        return data_gen, dataset_length

    if train:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=37129,
            batch_size=per_replica_train_batch_size,
            mode="train",
            data_split=0.8,
            device='gpu',
        )
    else:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=37129,
            batch_size=per_replica_val_batch_size,
            mode="val",
            data_split=0.8,
            device='gpu',
        )
    return data_gen, dataset_length