import os
import shutil
import numpy as np
import random
from ase.io import read
from ase import Atoms

# Global dictionary
length_dict = {'angstrom': 1.0, 'au': 0.5291772083}
energy_dict = {'au': 1.0, 'eV': 27.2114, 'kcal/mol': 627.509474}
force_dict = {'au/au': 0.529177, 'eV/angstrom': 27.2114}


# FUNCTIONS OF THE PROGRAM


def xyz_to_np(atoms_list: list[Atoms], saved_dir: str, unit_conversion: float = 1.0) -> None:
    r"""
    Convert *-pos-1.xyz file to 'coord.raw' file.

    :param atoms_list: A list consist of ase.Atoms that contains positions information.
    :param saved_dir: The saved directory of the 'coord.raw' file.
    :param unit_conversion: Convert the input length to angstrom, default conversion=1.0
    :return: None.
    """
    atom_n = len(atoms_list[0])
    total = np.empty((0, atom_n * 3), float)
    for atom in atoms_list:
        atom_n = len(atom)
        tmp = atom.get_positions()
        tmp = np.reshape(tmp, (1, atom_n * 3))
        total = np.concatenate((total, tmp), axis=0)
    total = total * unit_conversion
    saved_file = os.path.join(saved_dir, 'coord.npy')
    np.save(saved_file, total)


def energy_to_np(atoms_list: list[Atoms], saved_dir: str, unit_conversion: float = 1.0) -> None:
    r"""
    Convert *-pos-1.xyz file to 'energy.raw' file.

    :param atoms_list: A list consist of ase.Atoms that contains energy information.
    :param saved_dir: The saved directory of the 'coord.raw' file.
    :param unit_conversion: Convert the input energy to eV, default conversion=1.0.
    :return: None.
    """
    total = np.empty(0, float)
    for atom in atoms_list:
        tmp = atom.info['E']
        tmp = np.array(tmp, dtype="float")
        tmp = np.reshape(tmp, 1)
        total = np.concatenate((total, tmp), axis=0)
    total = total * unit_conversion
    saved_file = os.path.join(saved_dir, 'energy.npy')
    np.save(saved_file, total)


def force_to_np(frc_list: list[Atoms], saved_dir: str, unit_conversion: float = 1.0) -> None:
    r"""
    Convert *-frc-1.xyz file to 'energy.raw' file.

    :param frc_list: A list consist of ase.Atoms that contains energy information.
    :param saved_dir: The saved directory of the 'coord.raw' file.
    :param unit_conversion: Convert the input energy to eV, default conversion=1.0.
    :return: None.
    """
    atom_n = len(frc_list[0])
    total = np.empty((0, atom_n * 3), float)
    for atom in frc_list:
        atom_n = len(atom)
        tmp = atom.get_positions()
        tmp = np.reshape(tmp, (1, atom_n * 3))
        total = np.concatenate((total, tmp), axis=0)
    total = total * unit_conversion
    saved_file = os.path.join(saved_dir, 'force.npy')
    np.save(saved_file, total)


def read_cell(file_name: str) -> np.ndarray:
    r"""
    Read *.cell file to generate a ndarray containing cells information.

    :param file_name: The file to be read.
    :return: A ndarray with size of (num_cells, 9), which each row is the nine coordinate of the cell.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    cell_list = np.empty((0, 9), float)
    for line in lines:
        try:
            line_npy = np.array(line.rsplit()[2:11]).astype(float)
            line_npy = line_npy.reshape((1, 9))
            cell_list = np.concatenate((cell_list, line_npy), axis=0)
        except ValueError:
            pass
    return cell_list


def cell_to_np(atoms_list: list[Atoms], saved_dir: str, cell_len: float, unit_conversion: float = 1.0) -> None:
    r"""
    Convert single cubic cell to deepmd-kit compatible 'box.raw'.

    :param atoms_list: The corresponding atoms_list, mainly used for the length of the 'box.raw' file.
    :param saved_dir: The saved directory of the 'box.raw' file.
    :param cell_len: The single parameter for a cubic lattice constant.
    :param unit_conversion: Convert the input box length to angstrom, default conversion=1.0.
    :return: None.
    """
    total = np.empty((0, 9), float)
    frame_num = len(atoms_list)
    cells = np.array([[cell_len, 0, 0], [0, cell_len, 0], [0, 0, cell_len]], dtype="float")
    cells = np.reshape(cells, (1, 9))
    for frame in range(frame_num):
        total = np.concatenate((total, cells), axis=0)
    total = total * unit_conversion
    saved_file = os.path.join(saved_dir, 'box.npy')
    np.save(saved_file, total)


def cells_to_np(cells_array: np.ndarray, saved_dir: str, unit_conversion: float = 1.0) -> None:
    r"""

    :param cells_array:
    :param saved_dir:
    :param unit_conversion:
    :return:
    """
    saved_file = os.path.join(saved_dir, 'box.npy')
    total = cells_array * unit_conversion
    np.save(saved_file, total)


def type_raw(atoms_list: list[Atoms], saved_dir: str) -> None:
    r"""
    Convert atom type to the file 'type.raw' and 'type_map.raw'.

    :param atoms_list: A list of Atoms that contains the information of atom type.
    :param saved_dir: The directory that want to save the file.
    :return: None.
    """
    element = atoms_list[0].get_chemical_symbols()
    element = np.array(element)
    element, indices = np.unique(element, return_inverse=True)
    saved_file1 = os.path.join(saved_dir, 'type_map.raw')
    saved_file2 = os.path.join(saved_dir, 'type.raw')
    np.savetxt(saved_file1, element, fmt='%s', newline=' ')
    np.savetxt(saved_file2, indices, fmt='%s', newline=' ')


def cp2k2dpmd(save_dir: str, pos_file: str, cells: float or np.ndarray, length_unit: str = 'angstrom',
              ener_unit: str = 'eV', frc_file: str = None, frc_unit: str = None, valid_set_frac: float = 0.2) -> None:
    # Checking unit
    global length_dict, energy_dict, force_dict
    assert length_unit in length_dict.keys(), 'Not allowed length unit or misspell the word.'
    assert ener_unit in energy_dict.keys(), 'Not allowed energy unit or misspell the word.'
    if frc_file is not None:
        assert frc_unit in force_dict.keys(), 'Not allowed force unit or misspell the word.'
    # Generating saving directory
    save_dir = os.path.abspath(save_dir)
    data_0 = os.path.join(save_dir, 'data_0')
    data_0_set = os.path.join(data_0, 'set.000')
    data_1 = os.path.join(save_dir, 'data_1')
    data_1_set = os.path.join(data_1, 'set.000')
    for path in [data_0_set, data_0, data_1_set, data_1]:
        if os.path.isdir(path):
            shutil.rmtree(path)
    for path in [data_0, data_0_set, data_1, data_1_set]:
        os.mkdir(path)
    # Reading file
    atoms = read(pos_file, index=':')
    # Training & validation set split
    index = list(range(len(atoms)))
    random.shuffle(index)
    test_index = index[0:int(len(atoms)*valid_set_frac)+1]
    train_atoms = []
    test_atoms = []
    for i in range(len(atoms)):
        if i in test_index:
            test_atoms.append(atoms[i])
        else:
            train_atoms.append((atoms[i]))
    # Saving file:
    xyz_to_np(train_atoms, data_0_set, length_dict['angstrom']/length_dict[length_unit])
    xyz_to_np(test_atoms, data_1_set, length_dict['angstrom']/length_dict[length_unit])
    energy_to_np(train_atoms, data_0_set, energy_dict['eV']/energy_dict[ener_unit])
    energy_to_np(test_atoms, data_1_set, energy_dict['eV']/energy_dict[ener_unit])
    if frc_file is not None:
        forces = read(frc_file, index=':')
        train_force = []
        test_force = []
        for i in range(len(atoms)):
            if i in test_index:
                test_force.append(forces[i])
            else:
                train_force.append(forces[i])
        force_to_np(train_force, data_0_set, force_dict['eV/angstrom']/force_dict[frc_unit])
        force_to_np(test_force, data_1_set, force_dict['eV/angstrom']/force_dict[frc_unit])
    if isinstance(cells, float):
        cell_to_np(train_atoms, data_0_set, cells, length_dict['angstrom']/length_dict[length_unit])
        cell_to_np(test_atoms, data_1_set, cells, length_dict['angstrom']/length_dict[length_unit])
    elif isinstance(cells, np.ndarray):
        cells_to_np(cells, data_0_set, length_dict['angstrom']/length_dict[length_unit])
        cells_to_np(cells, data_1_set, length_dict['angstrom']/length_dict[length_unit])
    else:
        raise TypeError(f'Expect float or numpy.ndarray, but got {type(cells)} instead.')
    type_raw(atoms, data_0)
    type_raw(atoms, data_1)


def main():
    cells = read_cell('U_128-1.cell')
    cp2k2dpmd('./', 'U_128-pos-1.xyz', cells, ener_unit='au', frc_file='U_128-frc-1.xyz', frc_unit='au/au')


if __name__ == '__main__':
    main()
