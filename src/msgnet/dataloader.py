import os
import zlib
import pickle
import logging
import gzip
import tarfile
import io
import warnings
import requests
import numpy as np
import ase
import ase.db
from ase.neighborlist import NeighborList
from vorosym import voro_tessellate, graphdistance
import msgnet


class DataLoader:
    default_target = None
    default_datasplit_args = {"split_type": "fraction", "test_size": 0.1}

    def __init__(self):
        self.download_url = None
        self.download_dest = None
        self.cutoff_type = "const"
        self.cutoff_radius = 100.0
        self.self_interaction = False
        self.db_filter_query = None

    @property
    def final_dest(self):
        cutname = "%s-%.2f" % (self.cutoff_type, self.cutoff_radius)
        return "/%s/%s_%s.pkz" % (
            msgnet.defaults.datadir,
            self.__class__.__name__,
            cutname,
        )

    def _download_data(self):
        response = requests.get(self.download_url)
        with open(self.download_dest, "wb") as f:
            for chunk in response.iter_content():
                f.write(chunk)

    def _preprocess(self):
        graph_list = self.load_ase_data(
            db_path=self.download_dest,
            cutoff_type=self.cutoff_type,
            cutoff_radius=self.cutoff_radius,
            self_interaction=self.self_interaction,
            filter_query=self.db_filter_query,
        )
        return graph_list

    def _save(self, obj_list):
        with tarfile.open(self.final_dest, "w") as tar:
            for number, obj in enumerate(obj_list):
                pbytes = pickle.dumps(obj)
                cbytes = zlib.compress(pbytes)
                fsize = len(cbytes)
                cbuf = io.BytesIO(cbytes)
                cbuf.seek(0)
                tarinfo = tarfile.TarInfo(name="%d" % number)
                tarinfo.size = fsize
                tar.addfile(tarinfo, cbuf)

    def _load_data(self):
        obj_list = []
        with tarfile.open(self.final_dest, "r") as tar:
            for tarinfo in tar.getmembers():
                buf = tar.extractfile(tarinfo)
                decomp = zlib.decompress(buf)
                obj_list.append(pickle.loads(decomp))
        return obj_list

    @staticmethod
    def load_ase_data(
        db_path="oqmd_all_entries.db",
        dtype=float,
        cutoff_type="voronoi",
        cutoff_radius=2.0,
        filter_query=None,
        self_interaction=False,
        discard_unconnected=False,
    ):
        """load_ase_data
        Load atom structure data from ASE database

        :param db_path: path of the database to load
        :param dtype: dtype of returned numpy arrays
        :param cutoff_type: voronoi, const or coval
        :param cutoff_radius: cutoff radius of the sphere around each atom
        :param filter_query: query string or function to select a subset of database
        :param self_interaction: whether an atom includes itself as a neighbor (not only its images)
        :param discard_unconnected: whether to discard samples that ends up with no edges in the graph
        :return: list of FeatureGraph objects
        """
        con = ase.db.connect(db_path)
        sel = filter_query

        for i, row in enumerate(select_wfilter(con, sel)):
            if i % 100 == 0:
                print("%010d    " % i, sep="", end="\r")
            atoms = row.toatoms()
            if row.key_value_pairs:
                prop_dict = row.key_value_pairs
            else:
                prop_dict = row.data
            prop_dict["id"] = row.id
            try:
                graphobj = FeatureGraph(
                    atoms,
                    cutoff_type,
                    cutoff_radius,
                    lambda x: x,
                    self_interaction=self_interaction,
                    **prop_dict
                )
            except RuntimeError:
                logging.error("Error during data conversion of row id %d", row.id)
                continue
            if discard_unconnected and (graphobj.conns.shape[0] == 0):
                logging.error("Discarding %i because no connections made %s", i, atoms)
            else:
                yield graphobj
        print("")

    def load(self):
        if not os.path.isfile(self.final_dest):
            logging.info("%s does not exist" % self.final_dest)
            if not os.path.isfile(self.download_dest):
                logging.info(
                    "%s does not exist, downloading data..." % self.download_dest
                )
                self._download_data()
                logging.info("Download complete")
            logging.info("Preprocessing")
            obj_list = self._preprocess()
            logging.info("Saving to %s" % self.final_dest)
            self._save(obj_list)
            del obj_list
        logging.info("Loading data")
        obj_list = self._load_data()
        logging.info("Data loaded")
        return obj_list


class Oqmd12DataLoader(DataLoader):
    default_target = "delta_e"
    default_datasplit_args = {
        "split_type": "fold",
        "num_folds": 5,
        "validation_size": 5000,
    }

    def __init__(self, cutoff_type="voronoi", cutoff_radius=None):
        super().__init__()
        self.download_url = None
        self.download_dest = "%s/oqmd12.db" % (msgnet.defaults.datadir)
        self.cutoff_type = cutoff_type
        self.cutoff_radius = cutoff_radius
        self.self_interaction = False


class Oqmd12MiniDataLoader(DataLoader):
    default_target = "delta_e"
    default_datasplit_args = {
        "split_type": "fold",
        "num_folds": 5,
        "validation_size": 500,
    }

    def __init__(self, cutoff_type="voronoi", cutoff_radius=None):
        super().__init__()
        self.download_url = None
        self.download_dest = "%s/oqmd12.db" % (msgnet.defaults.datadir)
        self.cutoff_type = cutoff_type
        self.cutoff_radius = cutoff_radius
        self.self_interaction = False
        self.db_filter_query = "id<1000"


class MatprojDataLoader(DataLoader):
    default_target = "delta_e"
    default_datasplit_args = {
        "split_type": "fold",
        "num_folds": 5,
        "validation_size": 5000,
    }

    def __init__(self, cutoff_type="voronoi", cutoff_radius=None):
        super().__init__()
        self.download_url = None
        self.download_dest = "%s/matproj_2018.db" % (msgnet.defaults.datadir)
        self.cutoff_type = cutoff_type
        self.cutoff_radius = cutoff_radius
        self.self_interaction = False


class Qm9DataLoader(DataLoader):
    default_target = "U0"
    default_datasplit_args = {
        "split_type": "count",
        "validation_size": 10000,
        "test_size": 133885 - 120000,
    }

    def __init__(self, cutoff_type="const", cutoff_radius=100.0):
        super().__init__()
        self.download_url = None
        self.download_dest = "%s/qm9.db" % (msgnet.defaults.datadir)
        self.cutoff_type = cutoff_type
        self.cutoff_radius = cutoff_radius
        self.self_interaction = False


class FeatureGraph:
    def __init__(
        self,
        atoms_obj: ase.Atoms,
        cutoff_type,
        cutoff_radius,
        atom_to_node_fn,
        self_interaction=False,
        **kwargs
    ):
        self.atoms = atoms_obj

        if cutoff_type == "const":
            graph_tuple = self.atoms_to_graph_const_cutoff(
                self.atoms,
                cutoff_radius,
                atom_to_node_fn,
                self_interaction=self_interaction,
            )
            self.edge_labels = ["distance"]
        elif cutoff_type == "coval":
            graph_tuple = self.atoms_to_graph_const_cutoff(
                self.atoms,
                cutoff_radius,
                atom_to_node_fn,
                self_interaction=self_interaction,
                cutoff_covalent=True,
            )
            self.edge_labels = ["distance"]
        elif cutoff_type == "knearest":
            graph_tuple = self.atoms_to_graph_knearest(
                self.atoms, int(cutoff_radius), atom_to_node_fn
            )
            self.edge_labels = ["distance"]
        elif cutoff_type == "voronoi":
            graph_tuple = self.atoms_to_graph_voronoi(
                self.atoms,
                atom_to_node_fn,
                cutoff_radius,
                symmetry_binarize_threshold=0.99,
            )
            self.edge_labels = [
                "distance",
                "distance_normalized",
                "area",
                "area_normalized",
                "solid_angle",
                "C2",
                "C3",
                "C4",
                "C6",
                "D1",
                "D2",
                "D3",
                "D4",
                "D6",
            ]
        else:
            raise ValueError("cutoff_type not valid, given: %s" % cutoff_type)

        self.nodes, self.positions, self.edges, self.conns, self.conns_offset, self.unitcell = (
            graph_tuple
        )

        for key, val in kwargs.items():
            assert not hasattr(self, key), "Attribute %s is reserved" % key
            setattr(self, key, val)

    def remap_nodes(self, atom_to_node_fn):
        self.nodes = np.array(
            [atom_to_node_fn(n) for n in self.atoms.get_atomic_numbers()]
        )

    @staticmethod
    def atoms_to_graph_voronoi(
        atoms: ase.atoms,
        atom_to_node_fn,
        min_solid_angle=None,
        symmetry_binarize_threshold=None,
    ):

        nodes = []
        connections = []
        connections_offset = []
        edges = []

        voronoi_cells = voro_tessellate(atoms)

        assert np.all(
            atoms.get_pbc()
        ), "Voronoi graph only supported for periodic structures"
        atom_numbers = atoms.get_atomic_numbers()
        atom_positions = atoms.get_positions(wrap=True)
        unitcell = atoms.get_cell()
        for ii in range(len(atoms)):
            nodes.append(atom_to_node_fn(atom_numbers[ii]))

        for cell in voronoi_cells:
            total_area = sum(face.area for face in cell.faces)
            total_weighted_bond_length = sum(
                face.distance * face.solid_angle / (4 * np.pi) for face in cell.faces
            )
            # assert np.all(abs(atom_positions[i] - (cell.atom_pos - np.dot(cell.cell_offset, unitcell))) < 1e-4)
            for face in cell.faces:
                dist = face.distance
                area = face.area
                normed_area = face.area / total_area
                sangle = face.solid_angle
                if min_solid_angle and (sangle < min_solid_angle):
                    continue
                connections.append([face.neighbor, cell.atom_idx])  # [from, to]
                connections_offset.append(
                    np.vstack(
                        (face.neighbor_offset - cell.cell_offset, np.zeros(3, float))
                    )
                )
                if symmetry_binarize_threshold:
                    syms = [
                        float(x)
                        for x in (face.symmetries > symmetry_binarize_threshold)
                    ]
                else:
                    syms = list(face.symmetries)
                edges.append(
                    [dist, dist / total_weighted_bond_length, area, normed_area, sangle]
                    + syms
                )

        return (
            np.array(nodes),
            atom_positions,
            np.array(edges),
            np.array(connections),
            np.stack(connections_offset, axis=0),
            unitcell,
        )

    @staticmethod
    def atoms_to_graph_const_cutoff(
        atoms: ase.Atoms,
        cutoff,
        atom_to_node_fn,
        self_interaction=False,
        cutoff_covalent=False,
    ):

        atoms.wrap()
        atom_numbers = atoms.get_atomic_numbers()

        if cutoff_covalent:
            radii = ase.data.covalent_radii[atom_numbers] * cutoff
        else:
            radii = [cutoff] * len(atoms)
        neighborhood = NeighborList(
            radii, skin=0.0, self_interaction=self_interaction, bothways=True
        )
        neighborhood.update(atoms)

        nodes = []
        connections = []
        connections_offset = []
        edges = []
        if np.any(atoms.get_pbc()):
            atom_positions = atoms.get_positions(wrap=True)
        else:
            atom_positions = atoms.get_positions(wrap=False)
        unitcell = atoms.get_cell()

        for ii in range(len(atoms)):
            nodes.append(atom_to_node_fn(atom_numbers[ii]))

        for ii in range(len(atoms)):
            neighbor_indices, offset = neighborhood.get_neighbors(ii)
            for jj, offs in zip(neighbor_indices, offset):
                ii_pos = atom_positions[ii]
                jj_pos = atom_positions[jj] + np.dot(offs, unitcell)
                dist_vec = ii_pos - jj_pos
                dist = np.sqrt(np.dot(dist_vec, dist_vec))

                connections.append([jj, ii])
                connections_offset.append(np.vstack((offs, np.zeros(3, float))))
                edges.append([dist])

        if len(edges) == 0:
            warnings.warn("Generated graph has zero edges")
            edges = np.zeros((0, 1))
            connections = np.zeros((0, 2))
            connections_offset = np.zeros((0, 2, 3))
        else:
            connections_offset = np.stack(connections_offset, axis=0)

        return (
            np.array(nodes),
            atom_positions,
            np.array(edges),
            np.array(connections),
            connections_offset,
            unitcell,
        )

    @staticmethod
    def atoms_to_graph_knearest(
        atoms: ase.Atoms, num_neighbors, atom_to_node_fn, initial_radius=3.0
    ):

        atoms.wrap()
        atom_numbers = atoms.get_atomic_numbers()
        unitcell = atoms.get_cell()

        for multiplier in range(1, 11):
            if multiplier == 10:
                raise RuntimeError("Reached maximum radius")
            radii = [initial_radius * multiplier] * len(atoms)
            neighborhood = NeighborList(
                radii, skin=0.0, self_interaction=False, bothways=True
            )
            neighborhood.update(atoms)

            nodes = []
            connections = []
            connections_offset = []
            edges = []
            if np.any(atoms.get_pbc()):
                atom_positions = atoms.get_positions(wrap=True)
            else:
                atom_positions = atoms.get_positions(wrap=False)
            keep_connections = []
            keep_connections_offset = []
            keep_edges = []

            for ii in range(len(atoms)):
                nodes.append(atom_to_node_fn(atom_numbers[ii]))

            early_exit = False
            for ii in range(len(atoms)):
                this_edges = []
                this_connections = []
                this_connections_offset = []
                neighbor_indices, offset = neighborhood.get_neighbors(ii)
                if len(neighbor_indices) < num_neighbors:
                    # Not enough neigbors, so exit and increase radius
                    early_exit = True
                    break
                for jj, offs in zip(neighbor_indices, offset):
                    ii_pos = atom_positions[ii]
                    jj_pos = atom_positions[jj] + np.dot(offs, unitcell)
                    dist_vec = ii_pos - jj_pos
                    dist = np.sqrt(np.dot(dist_vec, dist_vec))

                    this_connections.append([jj, ii])  # from, to
                    this_connections_offset.append(
                        np.vstack((offs, np.zeros(3, float)))
                    )
                    this_edges.append([dist])
                edges.append(np.array(this_edges))
                connections.append(np.array(this_connections))
                connections_offset.append(np.stack(this_connections_offset, axis=0))
            if early_exit:
                continue
            else:
                for e, c, o in zip(edges, connections, connections_offset):
                    # Keep only num_neighbors closest indices
                    keep_ind = np.argsort(e[:, 0])[0:num_neighbors]
                    keep_edges.append(e[keep_ind])
                    keep_connections.append(c[keep_ind])
                    keep_connections_offset.append(o[keep_ind])
            break
        return (
            np.array(nodes),
            atom_positions,
            np.concatenate(keep_edges),
            np.concatenate(keep_connections),
            np.concatenate(keep_connections_offset),
            unitcell,
        )


def get_voro_adjacency(atoms: ase.Atoms, min_solid_angle=None):

    # Count how many times each atom is a neighbor of any other atom
    voronoi_cells = voro_tessellate(atoms)
    adjacency_count = np.zeros((len(atoms), len(atoms)), dtype=int)
    assert np.all(
        np.array([v.atom_idx for v in voronoi_cells]) == np.arange(len(atoms))
    )
    for cell in voronoi_cells:
        for face in cell.faces:
            if min_solid_angle and (face.solid_angle < min_solid_angle):
                continue
            adjacency_count[face.neighbor, cell.atom_idx] += 1

    # Convert adjacency count matrix to edge list
    edge_list = np.transpose(np.nonzero(adjacency_count)).astype(np.int32)
    graph_dist = graphdistance(edge_list, len(atoms))
    return adjacency_count, graph_dist


def select_wfilter(con, filterobj):
    if filterobj is None:
        for row in con.select():
            yield row
    elif isinstance(filterobj, str):
        for row in con.select(filterobj):
            yield row
    else:
        for row in con.select():
            if filterobj(row):
                yield row
