import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import msgnet
from tensorflow.contrib import layers


def compute_messages(
    nodes,
    conn,
    edges,
    message_fn,
    act_fn,
    include_receiver=True,
    include_sender=True,
    only_messages=False,
    mean_messages=True,
):
    """
    :param nodes: (n_nodes, n_node_features) tensor of nodes, float32.
    :param conn: (n_edges, 2) tensor of indices indicating an edge between nodes at those indices, [from, to] int32.
    :param edges: (n_edges, n_edge_features) tensor of edge features, float32.
    :param message_fn: message function, will be called with two inputs with shapes (n_edges, K*n_node_features), (n_edges,n_edge_features), where K is 2 if include_receiver=True and 1 otherwise, and must return a tensor of size (n_edges, n_output)
    :param act_fn: A pointwise activation function applied after the sum.
    :param include_receiver: Include receiver node in computation of messages.
    :param include_sender: Include sender node in computation of messages.
    :param only_messages: Do not sum up messages
    :param mean_messages: If true compute average over messages (instead of sum)
    :return: (n_edges, n_output) if only_messages is True, otherwise (n_nodes, n_output) Sum of messages arriving at each node.
    """
    n_nodes = tf.shape(nodes)[0]
    n_node_features = nodes.get_shape()[1].value
    n_edge_features = edges.get_shape()[1].value

    if include_receiver and include_sender:
        # Use both receiver and sender node features in message computation
        message_inputs = tf.gather(nodes, conn)  # n_edges, 2, n_node_features
        reshaped = tf.reshape(message_inputs, (-1, 2 * n_node_features))
    elif include_sender:  # Only use sender node features (index=0)
        message_inputs = tf.gather(nodes, conn[:, 0])  # n_edges, n_node_features
        reshaped = message_inputs
    elif include_receiver:  # Only use receiver node features (index=1)
        message_inputs = tf.gather(nodes, conn[:, 1])  # n_edges, n_node_features
        reshaped = message_inputs
    else:
        raise ValueError(
            "Messages must include at least one of sender and receiver nodes"
        )
    messages = message_fn(reshaped, edges)  # n_edges, n_output

    if only_messages:
        return messages

    idx_dest = conn[:, 1]
    if mean_messages:
        # tf.bincount not supported on GPU in TF 1.4, so do this instead
        count = tf.unsorted_segment_sum(
            tf.ones_like(idx_dest, dtype=tf.float32), idx_dest, n_nodes
        )
        count = tf.maximum(count, 1)  # Avoid division by zero
        msg_pool = tf.unsorted_segment_sum(
            messages, idx_dest, n_nodes
        ) / tf.expand_dims(count, -1)
    else:
        msg_pool = tf.unsorted_segment_sum(messages, idx_dest, n_nodes)
    return act_fn(msg_pool)


def create_dtnn_msg_function(num_outputs, num_hidden_neurons, **kwargs):
    """create_dtnn_msg_function
    Creates the message function from Deep Tensor Neural Networks (DTNN)

    :param num_outputs: output dimension
    :param num_hidden_neurons: number of hidden units
    :param **kwargs:
    """

    def func(nodes, edges):
        num_node_features = nodes.get_shape()[1].value
        num_edge_features = edges.get_shape()[1].value
        Wcf = tf.get_variable(
            "W_atom_c",
            (num_node_features, num_hidden_neurons),
            initializer=layers.xavier_initializer(False),
        )
        bcf = tf.get_variable(
            "b_atom_c", (num_hidden_neurons,), initializer=tf.constant_initializer(0)
        )
        Wdf = tf.get_variable(
            "W_dist",
            (num_edge_features, num_hidden_neurons),
            initializer=layers.xavier_initializer(False),
        )
        bdf = tf.get_variable(
            "b_dist", (num_hidden_neurons,), initializer=tf.constant_initializer(0)
        )
        Wfc = tf.get_variable(
            "W_hidden_to_c",
            (num_hidden_neurons, num_node_features),
            initializer=layers.xavier_initializer(False),
        )

        term1 = tf.matmul(nodes, Wcf) + bcf
        term2 = tf.matmul(edges, Wdf) + bdf
        output = tf.tanh(tf.matmul(term1 * term2, Wfc))
        return output

    return func


def create_msg_function(num_outputs, **kwargs):
    """create_msg_function
    Creates the message function used in the SchNet model

    :param num_outputs: number of output units
    :param **kwargs:
    """

    def func(nodes, edges):
        tf.add_to_collection("msg_input_nodes", nodes)
        tf.add_to_collection("msg_input_edges", edges)
        with tf.variable_scope("gates"):
            gates = msgnet.defaults.mlp(
                edges,
                [num_outputs, num_outputs],
                last_activation=msgnet.defaults.nonlinearity,
                activation=msgnet.defaults.nonlinearity,
                weights_initializer=msgnet.defaults.initializer,
            )
            tf.add_to_collection("msg_gates", gates)
        with tf.variable_scope("pre"):
            pre = layers.fully_connected(
                nodes,
                num_outputs,
                activation_fn=tf.identity,
                weights_initializer=msgnet.defaults.initializer,
                biases_initializer=None,
                **kwargs
            )
            tf.add_to_collection("msg_pregates", pre)
        output = pre * gates
        tf.add_to_collection("msg_outputs", output)
        return output

    return func


def edge_update(node_states, edge_states):
    """edge_update

    :param node_states: Tensor of dimension [number of nodes, node embedding size]
    :param edge_states: Tensor of dimension [number of edges, edge embedding size]
    """
    edge_states_len = int(edge_states.get_shape()[1])
    nodes_states_len = int(node_states.get_shape()[1])
    combined = tf.concat((node_states, edge_states), axis=1)
    new_edge = msgnet.defaults.mlp(
        combined,
        [nodes_states_len, nodes_states_len // 2],
        activation=msgnet.defaults.nonlinearity,
        weights_initializer=msgnet.defaults.initializer,
    )
    return new_edge


class MsgpassingNetwork:
    def __init__(
        self,
        n_node_features=1,
        n_edge_features=1,
        num_passes=3,
        embedding_shape=None,
        edge_feature_expand=None,
        msg_share_weights=False,
        use_edge_updates=False,
        readout_fn=None,
        edge_output_fn=None,
        avg_msg=False,
        target_mean=0.0,
        target_std=1.0,
    ):
        """__init__

        :param n_node_features: Number of input node features
        :param n_edge_features: Number of inpute edge features
        :param num_passes: Number of interaction pases
        :param embedding_shape: Shape of the atomic element embedding e.g. (num_species, embedding_size)
        :param edge_feature_expand: List of tuples for expanding edge features [(start, step, end)]
        :param msg_share_weights: Share weights between the interaction layers
        :param use_edge_updates: If true also update edges between interaction passes
        :param readout_fn: An instance of the ReadoutFunction class
        :param edge_output_fn: An instace of the EdgeOutputFunction class
        :param avg_msg: If true interaction messages will be averaged rather than summed
        :param target_mean: Normalization constant used for training on appropriate scale
        :oaram target_std: Normalization constant used for training on appropriate scale
        """

        # Symbolic input variables
        if embedding_shape is not None:
            self.sym_nodes = tf.placeholder(np.int32, shape=(None,), name="sym_nodes")
        else:
            self.sym_nodes = tf.placeholder(
                np.float32, shape=(None, n_node_features), name="sym_nodes"
            )
        self.sym_edges = tf.placeholder(
            np.float32, shape=(None, n_edge_features), name="sym_edges"
        )
        self.readout_fn = readout_fn
        self.edge_output_fn = edge_output_fn
        self.sym_conn = tf.placeholder(np.int32, shape=(None, 2), name="sym_conn")
        self.sym_segments = tf.placeholder(
            np.int32, shape=(None,), name="sym_segments_map"
        )
        self.sym_set_len = tf.placeholder(np.int32, shape=(None,), name="sym_set_len")

        self.input_symbols = {
            "nodes": self.sym_nodes,
            "edges": self.sym_edges,
            "connections": self.sym_conn,
            "segments": self.sym_segments,
            "set_lengths": self.sym_set_len,
        }

        # Setup constants for normalizing/denormalizing graph level outputs
        self.sym_target_mean = tf.get_variable(
            "target_mean",
            dtype=tf.float32,
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(target_mean),
        )
        self.sym_target_std = tf.get_variable(
            "target_std",
            dtype=tf.float32,
            shape=[],
            trainable=False,
            initializer=tf.constant_initializer(target_std),
        )

        if edge_feature_expand is not None:
            init_edges = msgnet.utilities.gaussian_expansion(
                self.sym_edges, edge_feature_expand
            )
        else:
            init_edges = self.sym_edges

        if embedding_shape is not None:
            # Setup embedding matrix
            stddev = np.sqrt(1.0 / np.sqrt(embedding_shape[1]))
            self.species_embedding = tf.Variable(
                initial_value=np.random.standard_normal(embedding_shape) * stddev,
                trainable=True,
                dtype=np.float32,
                name="species_embedding_matrix",
            )
            hidden_state0 = tf.gather(self.species_embedding, self.sym_nodes)
        else:
            hidden_state0 = self.sym_nodes

        hidden_state = hidden_state0

        hidden_state_len = int(hidden_state.get_shape()[1])

        # Setup edge update function
        if use_edge_updates:
            edge_msg_fn = edge_update
            edges = compute_messages(
                hidden_state,
                self.sym_conn,
                init_edges,
                edge_msg_fn,
                tf.identity,
                include_receiver=True,
                include_sender=True,
                only_messages=True,
            )
        else:
            edges = init_edges

        # Setup interaction messages
        msg_fn = create_msg_function(hidden_state_len)
        act_fn = tf.identity
        for i in range(num_passes):
            if msg_share_weights:
                scope_suffix = ""
                reuse = i > 0
            else:
                scope_suffix = "%d" % i
                reuse = False
            with tf.variable_scope("msg" + scope_suffix, reuse=reuse):
                sum_msg = compute_messages(
                    hidden_state,
                    self.sym_conn,
                    edges,
                    msg_fn,
                    act_fn,
                    include_receiver=False,
                    mean_messages=avg_msg,
                )
            with tf.variable_scope("update" + scope_suffix, reuse=reuse):
                hidden_state += msgnet.defaults.mlp(
                    sum_msg,
                    [hidden_state_len, hidden_state_len],
                    activation=msgnet.defaults.nonlinearity,
                    weights_initializer=msgnet.defaults.initializer,
                )
            with tf.variable_scope("edge_update" + scope_suffix, reuse=reuse):
                if use_edge_updates and (i < (num_passes - 1)):
                    edges = compute_messages(
                        hidden_state,
                        self.sym_conn,
                        edges,
                        edge_msg_fn,
                        tf.identity,
                        include_receiver=True,
                        include_sender=True,
                        only_messages=True,
                    )

            nodes_out = tf.identity(hidden_state, name="nodes_out")

        self.nodes_out = nodes_out

        # Setup readout function
        with tf.variable_scope("readout_edge"):
            if self.edge_output_fn is not None:
                self.edge_out = edge_output_fn(edges)
        with tf.variable_scope("readout_graph"):
            if self.readout_fn is not None:
                graph_out = self.readout_fn(nodes_out, self.sym_segments)
                self.graph_out_normalized = tf.identity(
                    graph_out, name="graph_out_normalized"
                )

        # Denormalize graph_out for making predictions on original scale
        if self.readout_fn is not None:
            if self.readout_fn.is_sum:
                mean = self.sym_target_mean * tf.expand_dims(
                    tf.cast(self.sym_set_len, tf.float32), -1
                )
            else:
                mean = self.sym_target_mean
            self.graph_out = tf.add(
                graph_out * self.sym_target_std, mean, name="graph_out"
            )

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=3)

    def save(self, session, destination, global_step):
        self.saver.save(session, destination, global_step=global_step)

    def load(self, session, path):
        self.saver.restore(session, path)

    def get_nodes_out(self):
        return self.nodes_out

    def get_graph_out(self):
        if self.readout_fn is None:
            raise NotImplementedError("No readout function given")
        return self.graph_out

    def get_graph_out_normalized(self):
        if self.readout_fn is None:
            raise NotImplementedError("No readout function given")
        return self.graph_out_normalized

    def get_normalization(self):
        return self.sym_target_mean, self.sym_target_std

    def get_readout_function(self):
        return self.readout_fn

    def get_edges_out(self):
        if self.edge_output_fn is None:
            raise NotImplementedError("No edges output network given")
        return self.edge_out

    def get_input_symbols(self):
        return self.input_symbols
