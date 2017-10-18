from scipy.sparse import *
import numpy as np
import os
import subprocess
from mlabwrap import mlab


def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
    A square matrix.

    Returns
    -------
    check : bool
    The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def get_hcp_resting(data_dir):
    """
    Return a list of addresses to data of each sessions of all subjects
    """
    subjects = os.listdir(data_dir)
    # HCP fix denoised resting state dataset
    sessions = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    # filenames is a list of address to individual subject (session if each subject has multiple sessions)
    filenames = [os.path.join(data_dir, subject, 'MNINonLinear/Results/%s/%s_Atlas_MSMAll_hp2000_clean.dtseries.nii'
                              % (session, session)) for subject in subjects for session in sessions]
    return filenames


def read_cifti_data(filename):
    """
    Read HCP CIFTI data using the workbench command tool
    """
    tmpfile = 'tmpfile_leica'
    subprocess.call(['wb_command', '-cifti-convert', '-to-text', filename, tmpfile])
    cii_data = np.loadtxt(tmpfile)
    subprocess.call(['rm', tmpfile])
    cii_data = cii_data[0:59412, :]  # select cortex data only
    return cii_data


def save_cifti_data(data, filename, wbc):
    """
    call the Matlab script to save to CIFTI format
    """
    mlab.myciftisave(data, filename, wbc)
