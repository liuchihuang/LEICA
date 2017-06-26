import os
import subprocess
import numpy as np
from scipy import sparse
from sklearn.preprocessing import scale
from sklearn.decomposition import FastICA
from data_handler import *
from utils import *


class LEICA(object):
    def __init__(self, data_dir, n_voxels, n_components, k, output_dir, batch_name=''):
        self.n_voxels = n_voxels
        self.n_components = n_components
        self.data_dir = data_dir
        self.subjects = os.listdir(self.data_dir)
        self.filenames = self.get_filenames()
        self.n_files = len(self.filenames)
        self.corr_matrix = np.zeros((self.n_voxels, self.n_voxels))
        self.k = k
        self.eigenmaps = None
        self.mask = None
        self.n_mask_voxels = self.n_voxels
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.independent_components = None

    # use different get functions for different data sets
    def get_filenames(self):
        filenames = get_hcp_resting(self.data_dir)
        return filenames

    def full_corr_matrix(self):
        print 'computing full correlation coefficient matrix'
        for filename in self.filenames:
            print '%s...' % filename,
            one_data = read_cifti_data(filename)  # for hcp cifti data
            self.corr_matrix += np.corrcoef(one_data)
            print 'done'
        self.corr_matrix /= self.n_files

    def remove_poor_connected_voxels(self):  # remove %1 poorest connected voxels
        degree = np.sum(self.corr_matrix, axis=0)
        degree_ind = np.argsort(degree)
        self.mask = [degree_ind[i] for i in range(self.n_voxels / 100, self.n_voxels)]
        self.corr_matrix = self.corr_matrix[self.mask, :]
        self.corr_matrix = self.corr_matrix[:, self.mask]
        self.n_mask_voxels = self.corr_matrix.shape[0]

    def knn_corr_matrix(self):
        self.remove_poor_connected_voxels()
        for i in range(self.n_mask_voxels):
            row = self.corr_matrix[i, :]
            row_sorted = np.sort(row)
            row[row < row_sorted[-self.k]] = 0
            self.corr_matrix[i, :] = row

        self.corr_matrix = sparse.csr_matrix(self.corr_matrix)

        corr_copy = self.corr_matrix.copy()
        corr_copy.data = np.ones(corr_copy.nnz)

        # either
        corr_copy += corr_copy.transpose()

        corr_copy.data = 1.0 / corr_copy.data

        self.corr_matrix += self.corr_matrix.transpose()
        self.corr_matrix = self.corr_matrix.multiply(corr_copy)

        self.corr_matrix.eliminate_zeros()

        assert is_symmetric(self.corr_matrix)
        print len(self.corr_matrix.data) / self.n_mask_voxels

    def laplacian_eigenmaps(self):
        degree = np.zeros(self.n_mask_voxels)
        degree_root = np.zeros(self.n_mask_voxels)

        for i in range(self.n_mask_voxels):
            degree[i] = sum(self.corr_matrix.data[self.corr_matrix.indptr[i]:self.corr_matrix.indptr[i + 1]])
            if degree[i] != 0:
                degree_root[i] = 1 / np.sqrt(degree[i])  # symmetric laplacian

        # compute D^(-1/2) W D^(-1/2)
        for i in range(self.n_mask_voxels):
            self.corr_matrix.data[self.corr_matrix.indptr[i]:self.corr_matrix.indptr[i + 1]] *= degree_root[i]
        self.corr_matrix = self.corr_matrix.tocoo()
        self.corr_matrix = self.corr_matrix.tocsc()
        for i in range(self.n_mask_voxels):
            self.corr_matrix.data[self.corr_matrix.indptr[i]:self.corr_matrix.indptr[i + 1]] *= degree_root[i]
        self.corr_matrix = self.corr_matrix.tocsr()

        print 'done'

        print 'computing eigen decomposition...',
        eigenvalues, self.eigenmaps = sparse.linalg.eigsh(self.corr_matrix, self.n_components+1, which='LA', tol=10**-10)
        self.eigenmaps = self.eigenmaps[:, -self.n_components - 1:-1]  # eliminate one evec
        print 'done'

        print eigenvalues

    def ica(self):
        print 'Starting ICA...'
        fica = FastICA(n_components=self.n_components, max_iter=10000)
        ic = fica.fit_transform(self.eigenmaps)
        ic = scale(ic, axis=0)
        # make components have positive peak
        for i in range(self.n_components):
            if max(ic[:, i]) + min(ic[:, i]) < 0:
                ic[:, i] = -ic[:, i]
        self.independent_components = np.zeros((self.n_voxels, self.n_components))
        self.independent_components[self.mask, :] = ic
        print self.independent_components.shape

    def save_outputs(self):
        np.save(os.path.join(self.output_dir, self.batch_name+'_leica.npy'), self.independent_components)
        save_cifti_data(self.independent_components, os.path.join(self.output_dir, self.batch_name+'_leica.dtseries.nii'))

        list_file = open(os.path.join(self.output_dir, '.list'), 'w')
        for i in range(self.n_components):
            list_file.write('%d\n' % (i+1))
        list_file.close()

        subprocess.call(['wb_command', '-cifti-convert-to-scalar', os.path.join(self.output_dir,
                        self.batch_name+'_leica.dtseries.nii'), 'ROW', os.path.join(self.output_dir,
                        self.batch_name+'_leica.dscalar.nii'), '-name-file', os.path.join(self.output_dir, '.list')])



