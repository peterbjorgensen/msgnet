import itertools
import math
import numpy as np
import sklearn.model_selection


def set_len_to_segments(set_len):
    return np.repeat(np.arange(len(set_len)), set_len)


def get_folds(objects, folds):
    return [o for o in objects if o.fold in folds]


class DataHandler:
    """DataHandler class used for handling and serving graph objects for training and testing"""

    def __init__(self, graph_objects, graph_targets=["total_energy"]):
        self.graph_objects = graph_objects
        self.graph_targets = graph_targets
        self.train_index_generator = self.idx_epoch_gen(len(graph_objects))

    def get_train_batch(self, batch_size):
        rand_choice = itertools.islice(self.train_index_generator, batch_size)
        training_dict = self.list_to_matrices(
            [self.graph_objects[idx] for idx in rand_choice],
            graph_targets=self.graph_targets,
        )
        self.modify_dict(training_dict)
        return training_dict

    def get_test_batches(self, batch_size):
        num_test_batches = int(math.ceil(len(self.graph_objects) / batch_size))
        for batch_idx in range(num_test_batches):
            test_dict = self.list_to_matrices(
                self.graph_objects[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ],
                graph_targets=self.graph_targets,
            )
            self.modify_dict(test_dict)
            yield test_dict

    def __len__(self):
        return len(self.graph_objects)

    @staticmethod
    def idx_epoch_gen(num_objects):
        while 1:
            for n in np.random.permutation(num_objects):
                yield n

    @staticmethod
    def list_to_matrices(graph_list, graph_targets=["total_energy"]):
        """list_to_matrices
        Convert list of FeatureGraph objects to dictionary with concatenated properties

        :param graph_list:
        :return: dictionary of stacked vectors and matrices
        """
        nodes_created = 0
        all_nodes = []
        all_conn = []
        all_conn_offsets = []
        all_edges = []
        all_graph_targets = []
        all_X = []
        all_unitcells = []
        set_len = []
        edges_len = []
        for gr in graph_list:
            nodes, conn, conn_offset, edges, X, unitcell = (
                gr.nodes,
                gr.conns,
                gr.conns_offset,
                gr.edges,
                gr.positions,
                gr.unitcell,
            )
            conn_shifted = np.copy(conn) + nodes_created
            all_nodes.append(nodes)
            all_conn.append(conn_shifted)
            all_conn_offsets.append(conn_offset)
            all_unitcells.append(unitcell)
            all_edges.append(edges)
            all_graph_targets.append(np.array([getattr(gr, t) for t in graph_targets]))
            all_X.append(X)
            nodes_created += nodes.shape[0]
            set_len.append(nodes.shape[0])
            edges_len.append(edges.shape[0])
        cat = lambda x: np.concatenate(x, axis=0)
        outdict = {
            "nodes": cat(all_nodes),
            "nodes_xyz": cat(all_X),
            "edges": cat(all_edges),
            "connections": cat(all_conn),
            "connections_offsets": cat(all_conn_offsets),
            "graph_targets": np.vstack(all_graph_targets),
            "set_lengths": np.array(set_len),
            "unitcells": np.stack(all_unitcells, axis=0),
            "edges_lengths": np.array(edges_len),
        }
        outdict["segments"] = set_len_to_segments(outdict["set_lengths"])
        return outdict

    def get_normalization(self, per_atom=False):
        x_sum = np.zeros(len(self.graph_targets))
        x_2 = np.zeros(len(self.graph_targets))
        num_objects = 0
        for obj in self.graph_objects:
            for i, target in enumerate(self.graph_targets):
                x = getattr(obj, target)
                if per_atom:
                    x = x / obj.nodes.shape[0]
                x_sum[i] += x
                x_2[i] += x ** 2.0
                num_objects += 1
        # Var(X) = E[X^2] - E[X]^2
        x_mean = x_sum / num_objects
        x_var = x_2 / num_objects - (x_mean) ** 2.0

        return x_mean, np.sqrt(x_var)

    def train_test_split(
        self,
        split_type=None,
        num_folds=None,
        test_fold=None,
        validation_size=None,
        test_size=None,
        deterministic=True,
    ):
        if split_type == "count" or split_type == "fraction":
            if deterministic:
                random_state = 21
            else:
                random_state = None
            if test_size > 0:
                train, test = sklearn.model_selection.train_test_split(
                    self.graph_objects, test_size=test_size, random_state=random_state
                )
            else:
                train = self.graph_objects
                test = []
        elif split_type == "fold":
            assert test_fold < num_folds
            assert test_fold >= 0
            train_folds = [i for i in range(num_folds) if i != test_fold]
            train, test = (
                get_folds(self.graph_objects, train_folds),
                get_folds(self.graph_objects, [test_fold]),
            )
        else:
            raise ValueError("Unknown split type %s" % split_type)

        if validation_size:
            if deterministic:
                random_state = 47
            else:
                random_state = None
            train, validation = sklearn.model_selection.train_test_split(
                train, test_size=validation_size, random_state=random_state
            )
        else:
            validation = []

        return self.from_self(train), self.from_self(test), self.from_self(validation)

    def modify_dict(self, train_dict):
        pass

    def from_self(self, objects):
        return self.__class__(objects, self.graph_targets)


class EdgeSelectDataHandler(DataHandler):
    """EdgeSelectDataHandler datahandler that selects a subset of the edge features"""

    def __init__(self, graph_objects, graph_targets, edge_input_idx):
        super().__init__(graph_objects, graph_targets)
        self.edge_input_idx = edge_input_idx

    def from_self(self, objects):
        return self.__class__(objects, self.graph_targets, self.edge_input_idx)

    def modify_dict(self, train_dict):
        all_edges = train_dict["edges"]
        input_edges = all_edges[:, self.edge_input_idx]
        train_dict["edges"] = input_edges


class EdgeOutDataHandler(DataHandler):
    """EdgeOutDataHandler datahandler that allows training with edge targets"""

    def __init__(self, graph_objects, graph_targets, edge_target_idx, edge_input_idx):
        super().__init__(graph_objects, graph_targets)
        self.edge_target_idx = edge_target_idx
        self.edge_input_idx = edge_input_idx

    def from_self(self, objects):
        return self.__class__(
            objects, self.graph_targets, self.edge_target_idx, self.edge_input_idx
        )

    def modify_dict(self, train_dict):
        all_edges = train_dict["edges"]
        target_edges = all_edges[:, self.edge_target_idx]
        input_edges = all_edges[:, self.edge_input_idx]
        train_dict["edges"] = input_edges
        train_dict["edges_targets"] = target_edges
