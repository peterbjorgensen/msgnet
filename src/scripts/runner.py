import sys
import os
import logging
import argparse
import timeit
import ase
import numpy as np
import tensorflow as tf

import msgnet


def list_of_lists(string):
    """list_of_list
    Parse a string of the form "1,2,3 4,3,2 ..."

    :param string:
    """
    if string:
        return [float(x) for x in string.strip().split(",")]
    else:
        return string


def float_or_string(string):
    try:
        return float(string)
    except ValueError:
        return string


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="@"
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument(
        "--cutoff",
        type=float_or_string,
        nargs="+",
        default=["voronoi"],
        help="Cutoff method (voronoi, const or coval) followed by float for const and coval",
    )
    parser.add_argument(
        "--edge_idx",
        nargs="*",
        default=[],
        type=int,
        help="Space separated list of edge feature indices to use",
    )
    parser.add_argument("--dataset", type=str, default="oqmd")
    parser.add_argument(
        "--edge_expand",
        nargs="*",
        type=list_of_lists,
        default=None,
        help="Space separated list of comma separated triplets start,step,stop for edge feature expansion",
    )
    parser.add_argument("--msg_share_weights", action="store_true")
    parser.add_argument("--num_passes", type=int, default=3)
    parser.add_argument("--node_embedding_size", type=int, default=256)
    parser.add_argument("--update_edges", action="store_true")
    parser.add_argument(
        "--avg_msg", action="store_true", help="Average messages instead of sum"
    )
    parser.add_argument("--readout", type=str, default="set2set")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    return parser.parse_args(arg_list)


def gen_prefix(namespace):
    prefix = []
    argdict = vars(namespace)
    for key in [
        "dataset",
        "cutoff",
        "edge_idx",
        "node_embedding_size",
        "msg_share_weights",
        "update_edges",
        "readout",
        "num_passes",
        "avg_msg",
        "fold",
        "target",
        "learning_rate",
    ]:
        if isinstance(argdict[key], list):
            val = "-".join([str(x) for x in argdict[key]])
        else:
            val = str(argdict[key])
        prefix.append(key[0] + val)
    return "_".join(prefix).replace(" ", "")


def get_dataloader_class(data_name):
    names = dir(msgnet.dataloader)
    lower_names = [x.lower() for x in names]
    index = lower_names.index(data_name + "dataloader")
    DataLoader = getattr(msgnet.dataloader, names[index])
    return DataLoader


def get_readout_function(readout_name, output_size):
    names = dir(msgnet.readout)
    lower_names = [x.lower() for x in names]
    index = lower_names.index("readout" + readout_name)
    ReadoutFunctionClass = getattr(msgnet.readout, names[index])
    return ReadoutFunctionClass(output_size)


def main(args):
    DataLoader = get_dataloader_class(args.dataset)

    if args.target:
        target_name = args.target
    else:
        target_name = DataLoader.default_target

    graph_obj_list = DataLoader(
        cutoff_type=args.cutoff[0], cutoff_radius=args.cutoff[1]
    ).load()
    data_handler = msgnet.datahandler.EdgeSelectDataHandler(
        graph_obj_list, [target_name], args.edge_idx
    )

    if not args.load_model:
        with open(logs_path + "commandline_args.txt", "w") as f:
            f.write("\n".join(sys.argv[1:]))

    train_obj, test_obj, val_obj = data_handler.train_test_split(
        test_fold=args.fold, **DataLoader.default_datasplit_args
    )

    model = get_model(args, data_handler)

    train_model(logs_path, model, args, target_name, train_obj, test_obj, val_obj)


def get_model(args, train_data_handler=None):
    readout_fn = get_readout_function(args.readout, 1)

    if train_data_handler:
        target_mean, target_std = train_data_handler.get_normalization(
            per_atom=readout_fn.is_sum
        )
    else:
        target_mean = 0
        target_std = 1
    logging.debug("Target mean %f, target std = %f" % (target_mean, target_std))

    net = msgnet.MsgpassingNetwork(
        n_node_features=1,
        n_edge_features=len(args.edge_idx),
        embedding_shape=(len(ase.data.chemical_symbols), args.node_embedding_size),
        edge_feature_expand=args.edge_expand,
        num_passes=args.num_passes,
        msg_share_weights=args.msg_share_weights,
        use_edge_updates=args.update_edges,
        readout_fn=readout_fn,
        avg_msg=args.avg_msg,
        target_mean=target_mean,
        target_std=target_std,
    )
    return net


def train_model(logs_path, model, args, target_name, train_obj, test_obj, val_obj=None):

    log_interval = len(train_obj)

    best_val_mae = np.inf
    best_val_step = 0

    # Write metadata for embedding visualisation
    with open(logs_path + "metadata.tsv", "w") as metaf:
        metaf.write("symbol\tnumber\t\n")
        for i, species in enumerate(ase.data.chemical_symbols):
            metaf.write("%s\t%d\n" % (species, i))
    with open(logs_path + "projector_config.pbtxt", "w") as logcfg:
        logcfg.write("embeddings {\n")
        logcfg.write(" tensor_name: 'species_embedding_matrix'\n")
        logcfg.write(" metadata_path: 'metadata.tsv'\n")
        logcfg.write("}")

    start_time = timeit.default_timer()
    logging.info("Training")
    num_steps = int(1e7)

    trainer = msgnet.train.GraphOutputTrainer(
        model, train_obj, initial_lr=args.learning_rate
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if args.load_model:
            if args.load_model.endswith(".meta"):
                checkpoint = args.load_model.replace(".meta", "")
                logging.info("loading model from %s", checkpoint)
                start_step = int(checkpoint.split("/")[-1].split("-")[-1])
                model.load(sess, checkpoint)
            else:
                checkpoint = tf.train.get_checkpoint_state(args.load_model)
                logging.info("loading model from %s", checkpoint)
                start_step = int(
                    checkpoint.model_checkpoint_path.split("/")[-1].split("-")[-1]
                )
                model.load(sess, checkpoint.model_checkpoint_path)
        else:
            start_step = 0

        # Print shape of all trainable variables
        trainable_vars = tf.trainable_variables()
        for var, val in zip(trainable_vars, sess.run(trainable_vars)):
            logging.debug("%s %s", var.name, var.get_shape())

        for update_step in range(start_step, num_steps):
            trainer.step(sess, update_step)

            if (update_step % log_interval == 0) or (update_step + 1) == num_steps:
                test_start_time = timeit.default_timer()

                # Evaluate training set
                train_metrics = trainer.evaluate_metrics(
                    sess, train_obj, prefix="train"
                )

                # Evaluate validation set
                if val_obj:
                    val_metrics = trainer.evaluate_metrics(sess, val_obj, prefix="val")
                else:
                    val_metrics = {}

                all_metrics = {**train_metrics, **val_metrics}
                metric_string = " ".join(
                    ["%s=%f" % (key, val) for key, val in all_metrics.items()]
                )

                end_time = timeit.default_timer()
                test_end_time = timeit.default_timer()
                logging.info(
                    "t=%.1f (%.1f) %d %s lr=%f",
                    end_time - start_time,
                    test_end_time - test_start_time,
                    update_step,
                    metric_string,
                    trainer.get_learning_rate(update_step),
                )
                start_time = timeit.default_timer()

                # Do early stopping using validation data (if available)
                if val_obj:
                    if all_metrics["val_mae"] < best_val_mae:
                        model.save(
                            sess, logs_path + "model.ckpt", global_step=update_step
                        )
                        best_val_mae = all_metrics["val_mae"]
                        best_val_step = update_step
                        logging.info(
                            "best_val_mae=%f, best_val_step=%d",
                            best_val_mae,
                            best_val_step,
                        )
                    if (update_step - best_val_step) > 1e6:
                        logging.info(
                            "best_val_mae=%f, best_val_step=%d",
                            best_val_mae,
                            best_val_step,
                        )
                        logging.info("No improvement in last 1e6 steps, stopping...")
                        model.save(
                            sess, logs_path + "model.ckpt", global_step=update_step
                        )
                        return
                else:
                    model.save(sess, logs_path + "model.ckpt", global_step=update_step)


if __name__ == "__main__":
    args = get_arguments()

    logs_path = "logs/runner_%s/" % gen_prefix(args)
    os.makedirs(logs_path, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(logs_path + "printlog.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )
    main(args)
