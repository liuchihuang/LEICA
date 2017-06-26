import os


def get_hcp_resting(data_dir):
    subjects = os.listdir(data_dir)
    # HCP fix denoised resting state dataset
    sessions = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    # filenames is a list of address to individual data (session if each subject has multiple sessions)
    filenames = [os.path.join(data_dir, subject, 'MNINonLinear/Results/%s/%s_Atlas_MSMAll_hp2000_clean.dtseries.nii'
                              % (sessions[i], sessions[i])) for subject in subjects for i in range(4)]
    return filenames
