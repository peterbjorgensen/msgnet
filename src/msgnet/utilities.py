import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
import math
import os
import scipy
import tensorflow as tf
import itertools


def tf_compute_distances(
    positions, unitcells, connections, connections_offset, segments
):
    """tf_compute_distances

    :param positions: (N_atoms, 3) tensor
    :param unitcells: (N_structures, 3, 3) tensor
    :param connections: (N_edges, 2) tensor
    :param connections_offset: (N_edges, 2, 3) tensor
    :param segments: N_atoms tensor
    """
    ## Compute absolute positions

    # Gather unit cells. We can assume that the unit cell is the same for sender and receiver
    unitcell_inds = tf.gather(segments, connections[:, 0])  # N_edges
    cells = tf.gather(unitcells, unitcell_inds)  # N_edges, 3, 3
    offsets = tf.matmul(connections_offset, cells)  # N_edges, 2, 3
    pos = tf.gather(positions, connections)  # N_edges, 2, 3
    abs_pos = pos + offsets
    diffs = abs_pos[:, 0, :] - abs_pos[:, 1, :]  # N_edges, 3
    dist = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=1, keepdims=True))  # N_edges, 3
    return dist


def entropy(p):
    """entropy
    Compute entropy of normalised discrete probability distribution
    :param p: (batch_size, Np) tensor
    """
    ent = tf.where(p > np.finfo(np.float32).eps, -p * tf.log(p), tf.zeros_like(p))
    ent = tf.reduce_sum(ent, axis=1)
    return ent


def gaussian_expansion(input_x, expand_params):
    """gaussian_expansion

    :param input_x: (num_edges, n_features) tensor
    :param expand_params: list of None or (start, step, stop) tuples
    :returns: (num_edges, ``ceil((stop - start)/step)``) tensor
    """
    feat_list = tf.unstack(input_x, axis=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            start, step, stop = step_tuple
            feat_expanded = tf.expand_dims(feat, axis=1)
            sigma = step
            mu = np.arange(start, stop, step)
            expanded_list.append(
                tf.exp(-((feat_expanded - mu) ** 2) / (2.0 * sigma ** 2))
            )
        else:
            expanded_list.append(tf.expand_dims(feat, 1))
    return tf.concat(expanded_list, axis=1, name="expanded_edges")
