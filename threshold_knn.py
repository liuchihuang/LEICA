import os
import sys
import numpy as np
from scipy import sparse
from utils import *
from scipy.sparse import diags, identity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale, normalize
from sklearn.utils import shuffle
from reprod_test import *
import timeit
import itertools
import subprocess


def construct_corr(subject_list):
    A_total = np.zeros((total_ordinates, total_ordinates))
    for sub_dir in subject_list:
        surf_data = []
        print 'loading %s...' % sub_dir,
        for n_run in run_list:
            one_run_data = read_cifti_data(os.path.join(data_dir, sub_dir, n_run))
            one_run_data = one_run_data[0:total_ordinates, :]
            one_run_data = scale(one_run_data, axis=1)

            surf_data.append(one_run_data)
        print 'done'
        surf_data = np.concatenate(surf_data, axis=1)

        print 'computing full correlation coefficient matrix'
        A_total += np.corrcoef(surf_data)
        print 'done'
    A_total /= len(subject_list)

    # print 'loading full correlation coefficient matrix'
    # A_total = np.load(os.path.join(inter_dir, 'similarity_matrix', 'A_full_100.npy'))
    # print 'done'

    # np.save(os.path.join(inter_dir, 'similarity_matrix', 'A_06_06.npy'), A_total)
    return A_total


def construct_norm_corr(subject_list):
    A_total = np.zeros((total_ordinates, total_ordinates))
    vars = np.zeros(total_ordinates)
    ts_length = 4800
    for sub_dir in subject_list:
        surf_data = []
        print 'loading %s...' % sub_dir,
        for n_run in run_list:
            one_run_data = read_norm_cifti_data(os.path.join(data_dir, sub_dir, n_run))
            surf_data.append(one_run_data)
        print 'done'
        surf_data = np.concatenate(surf_data, axis=1)

        print 'computing full correlation coefficient matrix'
        A_total += np.dot(surf_data, surf_data.T)
        vars += surf_data.var(axis=1)
        print 'done'
    vars = np.sqrt(ts_length * vars)
    vars = np.reshape(vars, (vars.shape[0], 1))
    A_total /= np.dot(vars, vars.T)
    print np.max(A_total)

    np.save(os.path.join(inter_dir, 'similarity_matrix', 'A_full_100_norm.npy'), A_total)
    return A_total


def eigenmap_th(A_total, ic_filename):
    degree = np.sum(A_total, axis=0)
    degree_ind = np.argsort(degree)

    ind_in = []
    ind_out = []
    for i in range(total_ordinates):
        if i <= total_ordinates / 100:
            ind_out.append(degree_ind[i])
        else:
            ind_in.append(degree_ind[i])

    A_total = A_total[ind_in, :]
    A_total = A_total[:, ind_in]

    total_ordinates_left = A_total.shape[0]

    k = 1200
    for i in range(total_ordinates_left):
        row = A_total[i, :]
        row_sorted = np.sort(row)
        row[row < row_sorted[-k]] = 0
        A_total[i, :] = row  #  the number of final neighbors

    A_total = sparse.csr_matrix(A_total)

    B = A_total.copy()

    for i in range(total_ordinates_left):
        row_len = len(A_total.data[A_total.indptr[i]:A_total.indptr[i + 1]])
        B.data[B.indptr[i]:B.indptr[i + 1]] = np.ones(row_len)

    # either
    B = B + B.transpose()
    for i in range(total_ordinates_left):
        B.data[B.indptr[i]:B.indptr[i + 1]] = 1.0 / B.data[B.indptr[i]:B.indptr[i + 1]]

    A_total += A_total.transpose()
    A_total = A_total.multiply(B)

    A_total.eliminate_zeros()

    assert is_symmetric(A_total)

    print len(A_total.data) / total_ordinates_left

    D = np.zeros(total_ordinates_left)
    D_sqrt = np.zeros(total_ordinates_left)

    for i in range(total_ordinates_left):
        D[i] = sum(A_total.data[A_total.indptr[i]:A_total.indptr[i + 1]])
        if D[i] != 0:
            D_sqrt[i] = 1 / np.sqrt(D[i])  # symmetric laplacian
            # D_sqrt[i] = 1 / D[i] # random walk laplacian

    # compute D^(-1/2) W D^(-1/2)
    for i in range(total_ordinates_left):
       A_total.data[A_total.indptr[i]:A_total.indptr[i + 1]] *= D_sqrt[i]
    A_total = A_total.tocoo()
    A_total = A_total.tocsc()
    for i in range(total_ordinates_left):
       A_total.data[A_total.indptr[i]:A_total.indptr[i + 1]] *= D_sqrt[i]
    A_total = A_total.tocsr()

    print 'done'

    print 'computing eigenvectors...'
    # L = I - D^(-1/2) W D^(-1/2), so SA becomes LA here
    evals, evecs = sparse.linalg.eigsh(A_total, n_components + 1, which='LA', tol=10**-10)
    print 'done'

    print evals
    # np.save(os.path.join(inter_dir, eig_filename), evecs)

    X = evecs[:, -n_components - 1:-1]  # eliminate one evec
    fica = FastICA(n_components=n_components, max_iter=1000)
    IC = fica.fit_transform(X).T
    print IC.shape
    print 'thresholding'
    IC = scale(IC, axis=1)
    tmp = np.zeros((n_components, total_ordinates))
    tmp[:, ind_in] = IC
    IC = tmp
    print IC.shape
    np.save('../result/'+ic_filename, IC)

    save_cifti_data(IC.T, os.path.join(result_dir, 'spec_knn.dtseries.nii'))
    IC = abs(IC)
    IC[IC < 2] = 0
    save_cifti_data(IC.T, os.path.join(result_dir, 'spec_knn_th.dtseries.nii'))
    subprocess.call(['wb_command', '-cifti-convert-to-scalar', os.path.join(result_dir, 'spec_knn_th.dtseries.nii'),
                     'ROW', os.path.join(result_dir, 'spec_knn_th.dscalar.nii'), '-name-file',
                     os.path.join(result_dir, 'list.txt')])


def rep_test(m):
    score = np.zeros(6)

    for pair in list(itertools.combinations(range(m),2)):
        ic1 = np.load('../result/' + filename(pair[0]))
        ic2 = np.load('../result/' + filename(pair[1]))
        score[0] += e_reprod(ic1, ic2)
        score[1] += t_reprod(ic1, ic2)

        ic1 = threshold_specica(ic1)
        ic2 = threshold_specica(ic2)
        score[2] += t_reprod(ic1, ic2)

    for i in range(m):
        ic1 = np.load('../result/' + filename(i))
        ic2 = np.load('../result/' + filename(-1))
        score[3] += e_reprod(ic1, ic2)
        score[4] += t_reprod(ic1, ic2)

        ic1 = threshold_specica(ic1)
        ic2 = threshold_specica(ic2)
        score[5] += t_reprod(ic1, ic2)
    score /= m
    print 'all scores:', score
    # record_file = open('../result/reproducibility.log', 'a')
    # record_file.write('%s\n' % score.tolist())
    # record_file.close()
    return score


# def sub_run():
#     record_file = open('../result/reproducibility.log', 'a')
#     record_file.write('=====new run=====\n')
#     record_file.close()
#
#     n_trials = 4
#     average_score = np.zeros((5, 6))
#     subject_list = os.listdir(surf_dir)
#
#     A_total1 = np.zeros((total_ordinates, total_ordinates))
#     A_total2 = np.zeros((total_ordinates, total_ordinates))
#     for trial in range(n_trials):
#         subject_list = shuffle(subject_list)
#         record_file = open('../result/reproducibility.log', 'a')
#         record_file.write('=====\n')
#         record_file.close()
#
#         for n_subgroup in range(5):
#             subgroup_list = subject_list[10 * n_subgroup: 10 * (n_subgroup + 1)]
#             A_total1 = (A_total1*n_subgroup * 10 + 10 * construct_corr(subgroup_list)) / ((n_subgroup + 1) * 10)
#             eigenmap_th(A_total1, ic1_filename)
#
#             subgroup_list = subject_list[50 + 10 * n_subgroup: 50 + 10 * (n_subgroup + 1)]
#             A_total2 = (A_total2 * n_subgroup * 10 + 10 * construct_corr(subgroup_list)) / ((n_subgroup + 1) * 10)
#             eigenmap_th(A_total2, ic2_filename)
#
#             average_score[n_subgroup, :] += rep_test(ic1_filename, ic2_filename)
#     average_score /= n_trials
#     record_file = open('../result/reproducibility.log', 'a')
#     record_file.write('=====\n')
#     for i in range(5):
#         record_file.write('%s\n' % average_score[i, :].tolist())
#     record_file.close()

def filename(m):
    if m == -1:
        return 'spec_th_knn_all.npy'
    else:
        return 'spec_th_knn%d_%d.npy' % (m, this_run)


def bootstrap():
    n_bootstrap = 3
    average_score = np.zeros((5, 6))
    subject_list = os.listdir(data_dir)

    all_sample_list = []
    for i in range(n_bootstrap):
        sample_index = np.random.choice(100, 50)
        one_sample_list = [subject_list[i] for i in sample_index]
        all_sample_list.append(one_sample_list)

    A_total_list = [np.zeros((total_ordinates, total_ordinates)) for i in range(n_bootstrap)]

    for n_subgroup in range(5):
        for i in range(n_bootstrap):
            subgroup_list = all_sample_list[i][10 * n_subgroup: 10 * (n_subgroup + 1)]
            A_total_list[i] = (A_total_list[i]*n_subgroup * 10 + 10 * construct_corr(subgroup_list)) / ((n_subgroup + 1) * 10)
            # A_total_list[i] = (A_total_list[i] * n_subgroup * 10 + 10 * construct_norm_corr(subgroup_list)) / ((n_subgroup + 1) * 10)
            eigenmap_th(A_total_list[i], filename(i))

        average_score[n_subgroup, :] += rep_test(n_bootstrap)

    record_file = open('../result/reproducibility.log', 'a')
    record_file.write('=====new bootstrapping k=1200 1%=====\n')
    for i in range(5):
        record_file.write('%d:%s\n' % ((i+1)*10, average_score[i, :].tolist()))
    record_file.close()

if __name__ == '__main__':
    start = timeit.default_timer()
    data_dir = '/fs/nara-scratch/HCP_S900_100unrelated_rsfMRI_fix/'
    inter_dir = '/fs/nara-scratch/chliu/fmri_proj/intermediate/'
    result_dir = '/fs/nara-scratch/chliu/fmri_proj/result/'


    run1 = 'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    run2 = 'MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    run3 = 'MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    run4 = 'MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    run_list = [run1, run2, run3, run4]

    eig_filename = 'eigenmap_th.npy'
    # ic1_filename = 'spec_th_knn1.npy'
    # ic2_filename = 'spec_th_knn2.npy'

    total_ordinates = 59412
    n_components = 20

    f = open(os.path.join('/fs/chib-ppmi/fmri_proj/similarity_matrix/', 'run_recorder.txt'), 'r+')
    this_run = int(f.read())
    f.seek(0, 0)
    f.write('%d\n' % (this_run+1))
    f.close()
    print 'THIS RUN:', this_run

    bootstrap()
    # sub_run()
    # allsub = os.listdir(surf_dir)
    # A = construct_corr(allsub[0:10])
    # eigenmap_th(A, 'spec_th_2000.npy')

    # construct_norm_corr(os.listdir(surf_dir))
    # A = np.load(os.path.join(inter_dir, 'similarity_matrix', 'A_full_100_norm.npy'))
    # eigenmap_th(A, 'spec_th_knn_norm.npy')

    stop = timeit.default_timer()
    print 'time consumed:', stop - start

