#!/usr/bin/env/ python3
import os
import numpy as np
import ase
import ase.db
import ase.data
import msgnet
import tarfile
import requests


def string_convert(string):
    try:
        return int(string)
    except ValueError:
        pass
    try:
        return float(string)
    except ValueError:
        pass
    return string.strip()


class Molecule:
    def __init__(self, num_atoms):
        self.z = np.zeros(num_atoms, dtype=np.int32)
        self.coord = np.zeros((num_atoms, 3))


def download(url, dest):
    response = requests.get(url)
    with open(dest, "wb") as f:
        for chunk in response.iter_content():
            f.write(chunk)


def tar_to_xyz(tarpath, dest):
    tar = tarfile.open(tarpath, mode="r:bz2")
    with open(os.path.join(msgnet.defaults.datadir, "qm9.xyz"), "wb") as f:
        for tarinfo in tar:
            tarf = tar.extractfile(tarinfo)
            f.write(tarf.read())


def load_xyz_file(filename):
    predefined_keys = """tag
    index
    A
    B
    C
    mu
    alpha
    homo
    lumo
    gap
    r2
    zpve
    U0
    U
    H
    G
    Cv""".split()
    STATE_READ_NUMBER = 0
    STATE_READ_COMMENT = 1
    STATE_READ_ENTRY = 2
    STATE_READ_FREQUENCY = 3
    STATE_READ_SMILES = 4
    STATE_READ_INCHI = 5
    STATE_FAILURE = 6

    state = STATE_READ_NUMBER
    entries_read = 0
    cur_desc = None

    with open(filename, "r") as f:
        for line_no, line in enumerate(f):
            try:
                if state == STATE_READ_NUMBER:
                    entries_to_read = int(line)
                    cur_desc = Molecule(entries_to_read)
                    entries_read = 0
                    state = STATE_READ_COMMENT
                elif state == STATE_READ_COMMENT:
                    # Read comment as whitespace separated values
                    for key, value in zip(predefined_keys, line.split()):
                        if hasattr(cur_desc, key):
                            raise KeyError(
                                "Molecule already contains property %s" % key
                            )
                        else:
                            setattr(cur_desc, key.strip(), string_convert(value))
                    state = STATE_READ_ENTRY
                elif state == STATE_READ_ENTRY:
                    parts = line.split()
                    assert len(parts) == 5
                    atom = parts[0]
                    el_number = ase.data.atomic_numbers[atom]
                    strat_parts = map(lambda x: x.replace("*^", "E"), parts[1:4])
                    floats = list(map(float, strat_parts))
                    cur_desc.coord[entries_read, :] = np.array(floats)
                    cur_desc.z[entries_read] = el_number
                    entries_read += 1
                    if entries_read == cur_desc.z.size:
                        state = STATE_READ_FREQUENCY
                elif state == STATE_READ_FREQUENCY:
                    cur_desc.frequency = np.array(
                        list(map(string_convert, line.split()))
                    )
                    state = STATE_READ_SMILES
                elif state == STATE_READ_SMILES:
                    cur_desc.smiles = line.split()
                    state = STATE_READ_INCHI
                elif state == STATE_READ_INCHI:
                    cur_desc.inchi = line.split()
                    yield cur_desc
                    state = STATE_READ_NUMBER
                elif state == STATE_FAILURE:
                    entries_to_read = None
                    try:
                        entries_to_read = int(line)
                    except:
                        pass
                    if entries_to_read is not None:
                        print("Resuming parsing on line %d" % line_no)
                        cur_desc = Molecule(entries_to_read)
                        entries_read = 0
                        state = STATE_READ_COMMENT
                else:
                    raise Exception("Invalid state")
            except Exception as e:
                print("Exception occured on line %d: %s" % (line_no, str(e)))
                state = STATE_FAILURE


def xyz_to_ase(filename, output_name):
    """
    Convert xyz descriptors to ase database
    """

    """
    =========================================================================================================
      Ele-    ZPVE         U (0 K)      U (298.15 K)    H (298.15 K)    G (298.15 K)     CV
      ment   Hartree       Hartree        Hartree         Hartree         Hartree        Cal/(Mol Kelvin)
    =========================================================================================================
       H     0.000000     -0.500273      -0.498857       -0.497912       -0.510927       2.981
       C     0.000000    -37.846772     -37.845355      -37.844411      -37.861317       2.981
       N     0.000000    -54.583861     -54.582445      -54.581501      -54.598897       2.981
       O     0.000000    -75.064579     -75.063163      -75.062219      -75.079532       2.981
       F     0.000000    -99.718730     -99.717314      -99.716370      -99.733544       2.981
    =========================================================================================================
    """
    HARTREE_TO_EV = 27.21138602
    REFERENCE_DICT = {
        ase.data.atomic_numbers["H"]: {
            "U0": -0.500273,
            "U": -0.498857,
            "H": -0.497912,
            "G": -0.510927,
        },
        ase.data.atomic_numbers["C"]: {
            "U0": -37.846772,
            "U": -37.845355,
            "H": -37.844411,
            "G": -37.861317,
        },
        ase.data.atomic_numbers["N"]: {
            "U0": -54.583861,
            "U": -54.582445,
            "H": -54.581501,
            "G": -54.598897,
        },
        ase.data.atomic_numbers["O"]: {
            "U0": -75.064579,
            "U": -75.063163,
            "H": -75.062219,
            "G": -75.079532,
        },
        ase.data.atomic_numbers["F"]: {
            "U0": -99.718730,
            "U": -99.717314,
            "H": -99.716370,
            "G": -99.733544,
        },
    }

    # Make a transposed dictionary such that first dimension is property
    REFERENCE_DICT_T = {}
    atom_nums = [ase.data.atomic_numbers[x] for x in ["H", "C", "N", "O", "F"]]
    for prop in ["U0", "U", "H", "G"]:
        prop_dict = dict(zip(atom_nums, [REFERENCE_DICT[at][prop] for at in atom_nums]))
        REFERENCE_DICT_T[prop] = prop_dict

    # List of tag, whether to convert hartree to eV
    keywords = [
        ["tag", False],
        ["index", False],
        ["A", False],
        ["B", False],
        ["C", False],
        ["mu", False],
        ["alpha", False],
        ["homo", True],
        ["lumo", True],
        ["gap", True],
        ["r2", False],
        ["zpve", True],
        ["U0", True],
        ["U", True],
        ["H", True],
        ["G", True],
        ["Cv", False],
    ]
    # Load xyz file
    descriptors = load_xyz_file(filename)

    with ase.db.connect(output_name, append=False) as asedb:
        properties_dict = {}
        for desc in descriptors:
            # Convert attributes to dictionary and convert hartree to eV
            for key, convert in keywords:
                properties_dict[key] = getattr(desc, key)
                # Subtract reference energies for each atom
                if key in REFERENCE_DICT_T:
                    for atom_num in desc.z:
                        properties_dict[key] -= REFERENCE_DICT_T[key][atom_num]
                if convert:
                    properties_dict[key] *= HARTREE_TO_EV
            atoms = ase.Atoms(numbers=desc.z, positions=desc.coord, pbc=False)
            asedb.write(atoms, data=properties_dict)


if __name__ == "__main__":
    url = "https://ndownloader.figshare.com/files/3195389"
    filename = os.path.join(msgnet.defaults.datadir, "dsgdb9nsd.xyz.tar.bz2")
    xyz_name = os.path.join(msgnet.defaults.datadir, "qm9.xyz")
    final_dest = os.path.join(msgnet.defaults.datadir, "qm9.db")
    print("downloading dataset...")
    download(url, filename)
    print("extracting...")
    tar_to_xyz(filename, xyz_name)
    print("writing to ASE database...")
    xyz_to_ase(xyz_name, final_dest)
    print("done")
