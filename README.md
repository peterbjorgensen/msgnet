# Msgnet
Tensorflow implementation of message passing neural networks for molecules and materials.
The framework implements the [SchNet model](https://arxiv.org/abs/1712.06113) and its extension with edge update network [NMP-EDGE](https://arxiv.org/abs/1806.03146) as well as the model used in [Materials property prediction using symmetry-labeled graphs as atomic-position independent descriptors](https://arxiv.org/abs/1905.06048).

Currently the implementation does not enable training with forces, but this might be implemented in the future.
For a more full-fledged implementation of the SchNet model, see [schnetpack](https://github.com/atomistic-machine-learning/schnetpack).

The main difference between `msgnet` and `schnetpack` is that `msgnet` follows a message passing architecture and can therefore be more flexible in some cases, e.g. it can be used to train on graphs rather than on structures with full spatial information.

# Install
Install the dependency
 - [Vorosym](https://github.com/peterbjorgensen/vorosym)

Set the `datadir` variable in `src/msgnet/defaults.py` to a preferred path in which the datasets will be saved.

Then run `python setup.py install` or `python setup.py install --user` to install the module.

## Install datasets 
Run the script `src/scripts/get_qm9.py` to download the QM9 dataset

Run the script `src/scripts/get_matproj.py MATPROJ_API_KEY` to download the materials project dataset. You need to create a user and obtain an API key from [Materials Project](https://materialsproject.org/).

Run the script `python2 src/scripts/get_oqmd.py` to convert the OQMD database into an ASE database. You need to manually download and install the [OQMD database](http://oqmd.org/) on your machine to run this script.
The OQMD API is only compatible with Python 2, so after running the script you must manually move the `oqmd12.db` to the `datadir` set in `src/msgnet/defaults.py`.

# Running the model
To train the model used in the NMP-EDGE paper:

`python runner.py  --cutoff const 100 --readout sumscalar  --num_passes 3 --update_edges --node_embedding_size 64 --dataset qm9  --edge_idx 0 --edge_expand 0.0,0.1,15.0  --learning_rate 5e-4 --target U0`

To train the model on OQMD structures using the voronoi graph with symmetry labels:
`python runner.py --fold 0 --cutoff voronoi 0.2 --readout avgscalar --num_passes 3 --node_embedding_size 256 --dataset oqmd12 --learning_rate 0.0001 --edge_idx 5 6 7 8 9 10 11 12 13 --update_edges`

After the model is done training get the test set results by running
`python predict_with_model --modelpath logs/path/to/model/model.ckpt-STEP.meta --output modeloutput.txt`

# Future Development
The model is implemented such that it avoids any padding/masking. This is achieved by reshaping the variable length inputs into the first dimension of the tensors, which is usually the batch dimension. However, this means we can't use the conventional Tensorflow methods for handling datasets as streams. If the framework is still used in the future I am planning to convert it into a tensorflow keras model when the RaggedTensor implementation is fully supported.
