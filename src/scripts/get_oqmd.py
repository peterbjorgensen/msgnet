#!/usr/bin/env/ python2
from __future__ import print_function
from __future__ import division
import pdb
import ase
import ase.db
import math
import qmpy
import sys
import subprocess
import numpy as np
from django.db.models import F

uniques = qmpy.Formation.objects.filter(entry__id=F("entry__duplicate_of__id"))
slice_size = 20000
db_path_tmp = "oqmd_tmp.db"
db_path_final = "oqmd12.db"


def write_slice(slice_id):
    if slice_id == 0:
        # We are writing a new database
        append = False
    else:
        # we are adding a slice to an existing database
        append = True
    with ase.db.connect(db_path_tmp, append=append) as ase_db:
        for i, formation in enumerate(
            uniques[(slice_id * slice_size) : ((slice_id + 1) * slice_size)]
        ):
            properties_dict = {}
            entry = formation.entry
            oqmd_id = entry.id
            # try:
            # spacegroup = entry.spacegroup.number
            # except AttributeError:
            # print("No spacegroup id for %d %d" % (i, oqmd_id))
            # spacegroup = "None"
            try:
                properties_dict["prototype"] = entry.prototype.name
            except AttributeError:
                pass

            try:
                atomic_numbers = entry.structure.atomic_numbers
                cell = entry.structure.cell
                cart_coords = entry.structure.cartesian_coords

                atoms = ase.Atoms(
                    positions=cart_coords, numbers=atomic_numbers, cell=cell, pbc=True
                )
            except Exception as e:
                pdb.set_trace()

            properties_dict["delta_e"] = entry.energy
            properties_dict["oqmd_id"] = oqmd_id
            if entry.label:
                properties_dict["label"] = entry.label

            ase_db.write(atoms, **properties_dict)


def generate_folds(num_entries, num_splits, ceil_first=True):
    """generate_folds

    :param num_entries:
    :param num_splits:
    :param ceil_first: if num_splits does not divide len(entry_list),
        first groups will be larger if ceil_first=True, otherwise they will be smaller
    """
    entries_left = num_entries
    fold_id = np.array([], dtype=int)
    for i in range(num_splits):
        if ceil_first:
            num_elements = int(math.ceil(entries_left / (num_splits - i)))
        else:
            num_elements = int(math.floor(entries_left / (num_splits - i)))
        print("fold %d: %d elements" % (i, num_elements))
        entries_left -= num_elements
        fold_id = np.concatenate([fold_id, np.ones(num_elements, dtype=int) * i])

    return np.random.permutation(fold_id)


def write_folds(input_db, output_db):
    # Set random seed
    np.random.seed(31)

    is_common_arr = []
    is_icsd_arr = []
    with ase.db.connect(input_db) as asedb:
        for row in asedb.select():
            is_common = True
            if row.delta_e > 5.0:
                is_common = False
            numbers = set(row.numbers)
            for element in ["He", "Ne", "Ar", "Kr", "Xe"]:
                if ase.atom.atomic_numbers[element] in numbers:
                    is_common = False
                    break
            try:
                label = row.label
                is_icsd = bool("icsd" in label.lower())
            except AttributeError:
                is_icsd = False
            is_common_arr.append(is_common)
            is_icsd_arr.append(is_icsd)

        is_common = np.array(is_common_arr, dtype=bool)
        is_icsd = np.array(is_icsd_arr, dtype=bool)
        folds = np.ones_like(is_common, dtype=int) * (-1)

        common_icsd = np.logical_and(is_common, is_icsd)
        common_nonicsd = np.logical_and(is_common, np.logical_not(is_icsd))

        icsd_folds = generate_folds(np.count_nonzero(common_icsd), 5, ceil_first=True)
        others_folds = generate_folds(
            np.count_nonzero(common_nonicsd), 5, ceil_first=False
        )
        folds[common_icsd] = icsd_folds
        folds[common_nonicsd] = others_folds

        with ase.db.connect(output_db, append=False) as foldsdb:
            for i, row in enumerate(asedb.select()):
                fold_id = folds[i]
                if fold_id < 0:
                    fold_name = "None"
                else:
                    fold_name = fold_id
                key_val_pairs = row.key_value_pairs
                key_val_pairs["fold"] = fold_name
                foldsdb.write(row.toatoms(), key_value_pairs=key_val_pairs)


def main():
    ## qmpy leaks memory (django cache?) so we need to process the database in slices
    try:
        slice_id = int(sys.argv[1])
        is_master = False
    except IndexError:
        is_master = True

    if is_master:
        num_slices = int(math.ceil(float(len(uniques)) / float(slice_size)))
        for slice_id in range(num_slices):
            subprocess.check_call(["python2", "get_oqmd.py", str(slice_id)])
            print("wrote %d/%d" % (slice_id + 1, num_slices))
        print("writing folds")
        write_folds(db_path_tmp, db_path_final)
    else:
        write_slice(slice_id)
        return 0


if __name__ == "__main__":
    main()
