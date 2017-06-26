import os
from mlabwrap import mlab

def get_hcp_resting(data_dir):
    subjects = os.listdir(data_dir)
    # HCP fix denoised resting state dataset
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    filenames = [os.path.join(data_dir, subject, 'MNINonLinear/Results/%s/%s_Atlas_MSMAll_hp2000_clean.dtseries.nii'
                              % (runs[i], runs[i])) for subject in subjects for i in range(4)]
    return filenames


def read_cifti_data(filename):
    wbc = '/fs/nara-scratch/chliu/fmri_proj/workbench/bin_rh_linux64/wb_command'
    cii_data = mlab.myciftiopen(filename, wbc)
    return cii_data


def save_cifti_data(data, filename):
    wbc = '/fs/nara-scratch/chliu/fmri_proj/workbench/bin_rh_linux64/wb_command'
    mlab.myciftisave(data, filename, wbc)