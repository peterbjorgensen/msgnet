#!/usr/bin/env/ python3
import math
import numpy as np
import pymatgen
import ase
import ase.db
import msgnet

# Available properties can be found here
# https://materialsproject.org/wiki/index.php/The_Materials_API#query


def get_all_results():
    with pymatgen.ext.matproj.MPRester(MATPROJ_API_KEY) as r:
        mp_ids = r.query({}, ["material_id"])
        material_ids = [m["material_id"] for m in mp_ids]
        print(len(material_ids))
        chunk_size = 1000
        sublists = [
            material_ids[i : i + chunk_size]
            for i in range(0, len(material_ids), chunk_size)
        ]
        for i, sublist in enumerate(sublists):
            results = r.query(
                {"material_id": {"$in": sublist}},
                [
                    "formation_energy_per_atom",
                    "final_structure",
                    "e_above_hull",
                    "material_id",
                    "icsd_id",
                    "band_gap",
                    "spacegroup",
                ],
            )
            print("Downloaded %d/%d" % (i + 1, len(sublists)))
            for res in results:
                yield res


def generate_folds(num_entries, num_splits, ceil_first=True, seed=42):
    """generate_folds

    :param num_entries:
    :param num_splits:
    :param ceil_first: if num_splits does not divide len(entry_list),
        first groups will be larger if ceil_first=True, otherwise they will be smaller
    """
    rng = np.random.RandomState(seed)
    entries_left = num_entries
    fold_id = np.array([], dtype=int)
    for i in range(num_splits):
        if ceil_first:
            num_elements = math.ceil(entries_left / (num_splits - i))
        else:
            num_elements = math.floor(entries_left / (num_splits - i))
        print("fold %d: %d elements" % (i, num_elements))
        entries_left -= num_elements
        fold_id = np.concatenate([fold_id, np.ones(num_elements, dtype=int) * i])

    return rng.permutation(fold_id)


def get_all_atoms():
    all_atoms = []
    is_common_arr = []
    for res in get_all_results():
        structure = res["final_structure"]
        delta_e = res["formation_energy_per_atom"]
        mp_id = res["material_id"]
        e_above_hull = res["e_above_hull"]
        icsd_id = res["icsd_id"]
        band_gap = res["band_gap"]
        sg_number = res["spacegroup"]["number"]
        sg_symbol = res["spacegroup"]["symbol"]
        sg_pointgroup = res["spacegroup"]["point_group"]
        sg_crystal_system = res["spacegroup"]["crystal_system"]

        cell = structure.lattice.matrix
        atomic_numbers = structure.atomic_numbers
        cart_coord = structure.cart_coords
        atoms = ase.Atoms(
            positions=cart_coord,
            numbers=atomic_numbers,
            cell=cell,
            pbc=[True, True, True],
        )

        # Check if material contains noble gas
        numbers = set(atoms.get_atomic_numbers())
        is_common = True
        for element in ["He", "Ne", "Ar", "Kr", "Xe"]:
            if ase.atom.atomic_numbers[element] in numbers:
                is_common = False
                break

        key_val_pairs = {
            "delta_e": delta_e,
            "mp_id": mp_id,
            "e_above_hull": e_above_hull,
            "band_gap": band_gap,
            "sg_number": sg_number,
            "sg_crystal_system": sg_crystal_system,
        }
        if icsd_id is not None:
            if isinstance(icsd_id, list):
                key_val_pairs["icsd_id"] = icsd_id[0]
            else:
                key_val_pairs["icsd_id"] = icsd_id

        all_atoms.append((atoms, key_val_pairs))
        is_common_arr.append(is_common)
    return all_atoms, is_common_arr


def main():
    all_atoms, is_common_arr = get_all_atoms()
    common_atoms = [a for a, c in zip(all_atoms, is_common_arr) if c]
    uncommon_atoms = [a for a, c in zip(all_atoms, is_common_arr) if not c]

    fold_id = generate_folds(len(common_atoms), 5)

    print("Writing to DB")
    with ase.db.connect(
        os.path.join(msgnet.defaults.datadir, "matproj.db"), append=False
    ) as db:
        for atom_keyval, fold in zip(common_atoms, fold_id):
            atom = atom_keyval[0]
            keyval = atom_keyval[1]
            keyval["fold"] = int(fold)
            db.write(atom, key_value_pairs=keyval)
        for atom, keyval in uncommon_atoms:
            keyval["fold"] = "None"
            db.write(atom, key_value_pairs=keyval)


if __name__ == "__main__":
    import sys

    try:
        MATPROJ_API_KEY = sys.argv[1]
    except IndexError:
        print("usage: python get_matproj.py MATPROJ_API_KEY")
    main()
