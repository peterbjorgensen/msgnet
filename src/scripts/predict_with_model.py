import math
import argparse
import os
import itertools
import logging

import numpy as np
import tensorflow as tf
import ase

import msgnet
import runner


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate graph convolution network")
    parser.add_argument("--modelpath", type=str, default=None)
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--cutoff",
        type=runner.float_or_string,
        nargs="+",
        default=[],
        help="Cutoff method (voronoi, const or coval) followed by float",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "validation", "all"], default="all"
    )
    return parser.parse_args()


class SpeciesPermutationsDataHandler(msgnet.datahandler.EdgeSelectDataHandler):
    def __init__(
        self,
        graph_objects,
        graph_targets,
        edge_input_idx,
        replace_species=None,
        keep_species=[],
    ):
        super().__init__(graph_objects, graph_targets, edge_input_idx)
        self.replace_species = replace_species
        self.keep_species = keep_species

    def from_self(self, objects):
        self.__class__(
            objects, self.graph_targets, self.replace_species, self.keep_species
        )

    def list_to_matrices(self, graph_list, graph_targets=["total_energy"]):
        """hack_list_to_matrices

        :param graph_list:
        :return: tuple of
            (nodes, conns, edges, node_targets, atom_pos, graph_target, set_len, edges_len)
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
            for rep_species in itertools.permutations(self.replace_species):
                nodes, conn, conn_offset, edges, X, unitcell = (
                    gr.nodes,
                    gr.conns,
                    gr.conns_offset,
                    gr.edges,
                    gr.positions,
                    gr.unitcell,
                )

                # Replace original species with given ones and ignore the keep_species
                to_be_replaced = set(nodes)
                to_be_replaced = to_be_replaced.difference(set(self.keep_species))
                if len(to_be_replaced) == 1:
                    rep = rep_species[0:1]
                else:
                    rep = rep_species
                    assert len(to_be_replaced) == len(rep_species)
                to_be_replaced = sorted(list(to_be_replaced))
                newnodes = nodes.copy()
                for spec, replacement in zip(to_be_replaced, rep):
                    newnodes[nodes == spec] = replacement
                nodes = newnodes

                conn_shifted = np.copy(conn) + nodes_created
                all_nodes.append(nodes)
                all_conn.append(conn_shifted)
                all_conn_offsets.append(conn_offset)
                all_unitcells.append(unitcell)
                all_edges.append(edges)
                all_graph_targets.append(
                    np.array([getattr(gr, t) for t in graph_targets])
                )
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
        outdict["segments"] = msgnet.datahandler.set_len_to_segments(
            outdict["set_lengths"]
        )
        return outdict


def species_count_filter(x, count):
    atm_nums = list(x.atoms.get_atomic_numbers())
    set_atm_nums = set(atm_nums)
    ordered_set = sorted(list(set_atm_nums))
    if not len(set_atm_nums) == count:
        return False
    return True


def si_filter(x):
    atm_nums = list(x.atoms.get_atomic_numbers())
    set_atm_nums = set(atm_nums)
    if len(set_atm_nums) > 1:
        return False
    if list(set_atm_nums)[0] == ase.data.atomic_numbers["Si"]:
        return True
    else:
        return False


def cu_filter(x):
    atm_nums = list(x.atoms.get_atomic_numbers())
    set_atm_nums = set(atm_nums)
    if len(set_atm_nums) > 1:
        return False
    if list(set_atm_nums)[0] == ase.data.atomic_numbers["Cu"]:
        return True
    else:
        return False


def zn_filter(x):
    atm_nums = list(x.atoms.get_atomic_numbers())
    set_atm_nums = set(atm_nums)
    if len(set_atm_nums) > 1:
        return False
    if list(set_atm_nums)[0] == ase.data.atomic_numbers["Zn"]:
        return True
    else:
        return False


def prototype_filter(x, prototype="Si(oS16)"):
    return x.prototype == prototype


def abse3_filter(x):
    atm_arr = x.atoms.get_atomic_numbers()
    atm_nums = list(atm_arr)
    set_atm_nums = set(atm_nums)
    if len(set_atm_nums) != 3:
        return False
    if ase.data.atomic_numbers["Se"] not in set_atm_nums:
        return False
    a, b = [c for c in list(set_atm_nums) if c != ase.data.atomic_numbers["Se"]]
    a_count = np.count_nonzero(a == atm_arr)
    b_count = np.count_nonzero(b == atm_arr)
    se_count = np.count_nonzero(ase.data.atomic_numbers["Se"] == atm_arr)
    if a_count == b_count and se_count / a_count == 3.0:
        return True
    else:
        return False


def icsd_filter(x):
    try:
        return "icsd" in x.label.lower()
    except AttributeError:
        return False


def unary_filter(x):
    return species_count_filter(x, 1)


def binary_filter(x):
    return species_count_filter(x, 2)


def ternary_filter(x):
    return species_count_filter(x, 3)


def icsd_unary_filter(x):
    return unary_filter(x) and icsd_filter(x)


def icsd_binary_filter(x):
    return binary_filter(x) and icsd_filter(x)


def icsd_ternary_filter(x):
    return ternary_filter(x) and icsd_filter(x)


def perovskite_filter(x):
    try:
        prototype = x.prototype
    except AttributeError:
        return False
    return prototype == "Perovskite"


def cu_prototype_filter(x):
    try:
        prototype = x.prototype
    except AttributeError:
        return False
    return prototype == "Cu"


def fe2p_filter(x):
    try:
        prototype = x.prototype
    except AttributeError:
        return False
    return prototype == "Fe2P"


def main():
    args = get_arguments()

    if args.filter:
        names = globals()
        filter_func = names[args.filter]
    else:
        filter_func = None

    metafile = args.modelpath
    checkpoint = metafile.replace(".meta", "")
    with open(
        os.path.join(os.path.dirname(metafile), "commandline_args.txt"), "r"
    ) as f:
        args_list = f.read().splitlines()
    runner_args = runner.get_arguments(args_list)

    if args.dataset:
        dataset = args.dataset
    else:
        dataset = runner_args.dataset
    if args.cutoff:
        cutoff = args.cutoff
    else:
        cutoff = runner_args.cutoff
    DataLoader = runner.get_dataloader_class(dataset)
    loader = DataLoader(cutoff_type=cutoff[0], cutoff_radius=cutoff[1])
    graph_obj_list = loader.load()
    if filter_func:
        graph_obj_list = [g for g in graph_obj_list if filter_func(g)]
    if args.permutations:
        graph_obj_list = [
            g for g in graph_obj_list if species_count_filter(g, args.permutations)
        ]
    if runner_args.target:
        target = runner_args.target
    else:
        target = loader.default_target

    if args.permutations:
        symbols = ["Ag", "C", "Na", "B", "Mg", "Cl"]
    else:
        symbols = ["Dummy"]
    with tf.Session() as sess:

        model = runner.get_model(runner_args)
        model.load(sess, checkpoint)

        for sym_i, symbol in enumerate(symbols):
            if args.permutations:
                if args.permutations == 1:
                    replace_species = [ase.data.atomic_numbers[symbols[sym_i]]]
                elif args.permutations == 2:
                    replace_species = [
                        ase.data.atomic_numbers[symbols[sym_i]],
                        ase.data.atomic_numbers[symbols[sym_i - 1]],
                    ]
                elif args.permutations == 3:
                    replace_species = [
                        ase.data.atomic_numbers[symbols[sym_i]],
                        ase.data.atomic_numbers[symbols[sym_i - 1]],
                        ase.data.atomic_numbers[symbols[sym_i - 2]],
                    ]
                else:
                    raise Exception("Invalid number of species to replace")
                keep_species = []
                datahandler = SpeciesPermutationsDataHandler(
                    graph_obj_list,
                    [target],
                    runner_args.edge_idx,
                    replace_species=replace_species,
                    keep_species=keep_species,
                )
            else:
                datahandler = msgnet.datahandler.EdgeSelectDataHandler(
                    graph_obj_list, [target], runner_args.edge_idx
                )

            if args.split in ["train", "test", "validation"]:
                if filter_func:
                    datasplit_args = DataLoader.default_datasplit_args
                    datasplit_args["validation_size"] = 0
                else:
                    datasplit_args = DataLoader.default_datasplit_args
                splits = dict(
                    zip(
                        ["train", "test", "validation"],
                        datahandler.train_test_split(
                            **DataLoader.default_datasplit_args,
                            test_fold=runner_args.fold
                        ),
                    )
                )
                datahandler = splits[args.split]

            target_values = np.array(
                [getattr(g, target) for g in datahandler.graph_objects]
            )
            row_id = np.array([g.id for g in datahandler.graph_objects])

            if args.permutations:
                repeats = math.factorial(args.permutations)
                target_values = np.repeat(target_values, repeats)
                row_id = np.repeat(row_id, repeats)

            model_predictions = []
            print("computing predictions")
            for input_data in datahandler.get_test_batches(5):
                feed_dict = {}
                model_input_symbols = model.get_input_symbols()
                for key, val in model_input_symbols.items():
                    feed_dict[val] = input_data[key]
                graph_out, = sess.run([model.get_graph_out()], feed_dict=feed_dict)
                model_predictions.append(graph_out)

            model_predict = np.concatenate(model_predictions, axis=0).squeeze()

            if args.permutations:
                outpath = "%s_%s" % (args.output, symbol)
            else:
                outpath = args.output

            errors = model_predict - target_values
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(np.square(errors)))

            print("split=%s, num_samples=%d, mae=%s, rmse=%s" % (args.split, errors.shape[0], mae, rmse))

            np.savetxt(
                outpath, np.stack((target_values, model_predict, row_id), axis=1)
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[logging.StreamHandler()],
    )

    main()
