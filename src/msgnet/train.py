import tensorflow as tf
import numpy as np
import msgnet


class Trainer:
    def __init__(self, model, batchloader, initial_lr=1e-4, batch_size=32):
        self.model = model
        self.sym_learning_rate = tf.placeholder(
            tf.float32, shape=[], name="learning_rate"
        )

        self.initial_lr = initial_lr

        self.input_symbols = self.setup_input_symbols()
        self.cost = self.setup_total_cost()
        self.train_op = self.setup_train_op()
        self.metric_tensors = self.setup_metrics()
        self.batchloader = batchloader
        self.batch_size = batch_size

    def get_learning_rate(self, step):
        learning_rate = self.initial_lr * (0.96 ** (step / 100000))
        return learning_rate

    def setup_metrics(self):
        return {}

    def setup_input_symbols(self):
        input_symbols = self.model.get_input_symbols()
        return input_symbols

    def setup_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.sym_learning_rate)
        gradients = optimizer.compute_gradients(self.cost)
        train_op = optimizer.apply_gradients(gradients, name="train_op")
        return train_op

    def setup_total_cost(self):
        raise NotImplementedError()

    def step(self, session, step):
        input_data = self.batchloader.get_train_batch(self.batch_size)
        feed_dict = {}
        for key in self.input_symbols.keys():
            feed_dict[self.input_symbols[key]] = input_data[key]
        feed_dict[self.sym_learning_rate] = self.get_learning_rate(step)
        session.run([self.train_op], feed_dict=feed_dict)


class GraphOutputTrainer(Trainer):
    def setup_input_symbols(self):
        input_symbols = self.model.get_input_symbols()
        output_size = self.model.get_graph_out().get_shape()[1].value

        self.sym_edge_targets = tf.placeholder(
            tf.float32, shape=(None, 1), name="sym_edge_targets"
        )
        self.sym_graph_targets = tf.placeholder(
            tf.float32, shape=(None, output_size), name="sym_graph_targets"
        )

        input_symbols.update({"graph_targets": self.sym_graph_targets})

        return input_symbols

    def setup_metrics(self):
        graph_error = self.model.get_graph_out() - self.input_symbols["graph_targets"]

        metric_tensors = {"graph_error": graph_error}
        return metric_tensors

    def setup_total_cost(self):
        sym_graph_targets = self.input_symbols["graph_targets"]
        graph_cost = self.get_cost_graph_target(sym_graph_targets, self.model)
        total_cost = graph_cost
        return total_cost

    def evaluate_metrics(self, session, datahandler, prefix=""):
        target_mae = 0
        target_mse = 0
        num_graphs = 0
        for input_data in datahandler.get_test_batches(self.batch_size):
            feed_dict = {}
            for key in self.input_symbols.keys():
                feed_dict[self.input_symbols[key]] = input_data[key]
            syms = [self.metric_tensors["graph_error"]]
            graph_error, = session.run(syms, feed_dict=feed_dict)
            target_mae += np.sum(np.abs(graph_error))
            target_mse += np.sum(np.square(graph_error))
            num_graphs += graph_error.shape[0]

        if prefix:
            prefix += "_"
        metrics = {
            prefix + "mae": target_mae / num_graphs,
            prefix + "rmse": np.sqrt(target_mse / num_graphs),
        }

        return metrics

    @staticmethod
    def get_cost_graph_target(sym_graph_target, model):
        target_mean, target_std = model.get_normalization()
        sym_set_len = model.get_input_symbols()["set_lengths"]
        target_normalizing = 1.0 / target_std
        if model.get_readout_function().is_sum:
            # When target is a sum of K numbers we normalize the target to zero mean and variance K
            graph_target_normalized = (
                sym_graph_target
                - target_mean * tf.cast(tf.expand_dims(sym_set_len, -1), tf.float32)
            ) * target_normalizing
        else:
            # When target is an average normalize to zero mean and unit variance
            graph_target_normalized = (
                sym_graph_target - target_mean
            ) * target_normalizing

        graph_cost = tf.reduce_mean(
            (model.get_graph_out_normalized() - graph_target_normalized) ** 2,
            name="graph_cost",
        )

        return graph_cost


class EdgeOutputTrainer(GraphOutputTrainer):
    def __init__(
        self,
        model,
        batchloader,
        edge_output_expand,
        initial_lr=1e-4,
        edge_cost_weight=0.5,
    ):
        self.edge_cost_weight = edge_cost_weight
        self.edge_output_expand = edge_output_expand

        super().__init__(model, batchloader, initial_lr=initial_lr)

    def setup_input_symbols(self):
        input_symbols = self.model.get_input_symbols()
        output_size = self.model.get_graph_out().get_shape()[1].value

        self.sym_edge_targets = tf.placeholder(
            tf.float32, shape=(None, 1), name="sym_edge_targets"
        )
        self.sym_graph_targets = tf.placeholder(
            tf.float32, shape=(None, output_size), name="sym_graph_targets"
        )

        input_symbols.update(
            {
                "edges_targets": self.sym_edge_targets,
                "graph_targets": self.sym_graph_targets,
            }
        )

        return input_symbols

    def setup_metrics(self):
        graph_error = self.model.get_graph_out() - self.input_symbols["graph_targets"]
        sym_edge_targets = self.input_symbols["edges_targets"]
        edge_error = self.get_cost_edge_target(sym_edge_targets, self.model)

        metric_tensors = {"graph_error": graph_error, "edge_error": edge_error}
        return metric_tensors

    def setup_total_cost(self):
        sym_graph_targets = self.input_symbols["graph_targets"]
        sym_edge_targets = self.input_symbols["edges_targets"]
        graph_cost = self.get_cost_graph_target(sym_graph_targets, self.model)
        edge_cost = tf.reduce_mean(
            self.get_cost_edge_target(sym_edge_targets, self.model), name="edge_cost"
        )
        total_cost = (
            1 - self.edge_cost_weight
        ) * graph_cost + self.edge_cost_weight * edge_cost
        return total_cost

    def evaluate_metrics(self, session, datahandler, prefix=""):
        target_mae = 0
        target_mse = 0
        edge_kl = 0
        num_graphs = 0
        num_edges = 0
        for input_data in datahandler.get_test_batches(self.batch_size):
            feed_dict = {}
            for key in self.input_symbols.keys():
                feed_dict[self.input_symbols[key]] = input_data[key]
            syms = [
                self.metric_tensors["graph_error"],
                self.metric_tensors["edge_error"],
            ]
            graph_error, edge_error = session.run(syms, feed_dict=feed_dict)
            edge_kl += np.sum(edge_error)
            target_mae += np.sum(np.abs(graph_error))
            target_mse += np.sum(np.square(graph_error))
            num_graphs += graph_error.shape[0]
            num_edges += edge_error.shape[0]

        if prefix:
            prefix += "_"
        metrics = {
            prefix + "mae": target_mae / num_graphs,
            prefix + "rmse": np.sqrt(target_mse / num_graphs),
            prefix + "kl": edge_kl / num_edges,
        }

        return metrics

    def get_cost_edge_target(self, sym_edge_target, model):
        edge_expanded = msgnet.utilities.gaussian_expansion(
            sym_edge_target, self.edge_output_expand
        )
        edge_expanded_normalised = tf.divide(
            edge_expanded, tf.reduce_sum(edge_expanded, axis=1, keepdims=True)
        )
        p_entropy = msgnet.utilities.entropy(edge_expanded_normalised)
        edge_cost = (
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=edge_expanded_normalised, logits=model.get_edges_out()
            )
            - p_entropy
        )

        return edge_cost
