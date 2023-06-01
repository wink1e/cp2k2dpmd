import glob
import os
import shutil
import numpy as np
import random
from ase.io import read
from ase import Atoms

# DATA IMPORT

data_path = "C:\\Users\\12429\\Desktop\\work\\Am1"  # Windows用\\,linux系统下用/
atom_num = None
au_to_eV = 2.72113838565563E+01
au_to_A = 5.29177208590000E-01
au_to_kcal = 6.2750947415E+02


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
    saved_file = os.path.join(saved_dir, 'coord.raw')
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
        tmp = list(atom.info.keys())
        if "Energy:" in tmp:
            tmp.remove("Energy:")
        tmp = np.array(tmp, dtype="float")
        tmp = np.reshape(tmp, 1)
        total = np.concatenate((total, tmp), axis=0)
    total = total * unit_conversion
    saved_file = os.path.join(saved_dir, 'energy.raw')
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
    saved_file = os.path.join(saved_dir, 'force.raw')
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


# TODO
def cell_to_np(atoms_list: list[Atoms], saved_dir: str, cell_len: float, unit_conversion: float = 1.0):
    total = np.empty((0, 9), float)
    frame_num = len(atoms_list)
    cells = np.array([[cell_len, 0, 0], [0, cell_len, 0], [0, 0, cell_len]], dtype="float")
    cells = np.reshape(cells, (1, 9))
    for frame in range(frame_num):
        total = np.concatenate((total, cells), axis=0)
    total = total * unit_conversion
    saved_file = os.path.join(saved_dir, 'box.raw')
    np.save(saved_file, total)


# TODO
def cells_to_np(cells_array: np.ndarray, saved_dir: str, unit_conversion: float = 1.0):
    pass


# TODO
def type_raw(position, output):
    element = position.get_chemical_symbols()
    element = np.array(element)
    tmp, indices = np.unique(element, return_inverse=True)
    np.savetxt(output, indices, fmt='%s', newline=' ')


def cp2k2dpmd(pos_file: str, frc_file: str = None, ):
    pass


# MAIN PROGRAM

# data_path = os.path.abspath(data_path)
# pos_path = os.path.join(data_path, "Am1_*")
# frc_path = os.path.join(data_path, "*frc-1.xyz")
# pos_path = glob.iglob(pos_path)

# for po in pos_path:
#     file_path = os.path.join(po, "*.xyz")
#     set_path_1 = os.path.join(po, "data_0")
#     set_path_2 = os.path.join(set_path_1, "set.000")
#     set_path_3 = os.path.join(po, "data_1")
#     set_path_4 = os.path.join(set_path_3, "set.000")
#     file_path = glob.iglob(file_path)
#     saved_pos = []
#     atom_num = None
#     for file in file_path:
#         temp = read(file, index=0)
#         ind = temp.get_atomic_numbers()
#         ind = ind == 1
#         del temp[ind]
#         if atom_num is None:
#             atom_num = temp.get_global_number_of_atoms()
#         saved_pos.append(temp)
#     train_pos = []
#     valid_pos = []
#     for po in saved_pos:
#         randflag = random.random()
#         if randflag < 0.8:
#             train_pos.append(po)
#         else:
#             valid_pos.append(po)
#
#     if os.path.isdir(set_path_2):
#         shutil.rmtree(set_path_2)
#     if os.path.isdir(set_path_1):
#         shutil.rmtree(set_path_1)
#     if os.path.isdir(set_path_3):
#         shutil.rmtree(set_path_3)
#     if os.path.isdir(set_path_4):
#         shutil.rmtree(set_path_4)
#     os.mkdir(set_path_1)
#     os.mkdir(set_path_2)
#     os.mkdir(set_path_3)
#     os.mkdir(set_path_4)
#     type_path = os.path.join(set_path_1, "type.raw")
#     coord_path = os.path.join(set_path_2, "coord.npy")
#     box_path = os.path.join(set_path_2, "box.npy")
#     energy_path = os.path.join(set_path_2, "energy.npy")
#     type_path_v = os.path.join(set_path_3, "type.raw")
#     coord_path_v = os.path.join(set_path_4, "coord.npy")
#     box_path_v = os.path.join(set_path_4, "box.npy")
#     energy_path_v = os.path.join(set_path_4, "energy.npy")
#
#     xyz_to_np(train_pos, atom_num, coord_path)
#     energy_to_np(train_pos, energy_path, au_to_eV / au_to_kcal)
#     cell_to_np(train_pos, box_path, cell)
#     type_raw(train_pos[0], type_path)
#
#     xyz_to_np(valid_pos, atom_num, coord_path_v)
#     energy_to_np(valid_pos, energy_path_v, au_to_eV / au_to_kcal)
#     cell_to_np(valid_pos, box_path_v, cell)
#     type_raw(valid_pos[0], type_path_v)

if __name__ == '__main__':
    atom_list = read('U_128-pos-1.xyz', index=':')
    print(len(atom_list[0]))
