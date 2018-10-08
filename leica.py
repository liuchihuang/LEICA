##############################################################################
#
#  LEICA: Laplacian Eigenmaps for group ICA Decomposition of fMRI data
#
##############################################################################

import os
import subprocess
import numpy as np
from scipy import sparse
from sklearn.preprocessing import scale
from sklearn.decomposition import FastICA
from utils import *


class LEICA(object):
    def __init__(self, wbc, data_dir, output_dir=None, n_components=20, k=None, batch_name=''):
        self.n_voxels = None
        self.n_components = n_components
        self.data_dir = data_dir
        self.filenames = None
        self.n_files = None
        self.corr_matrix = None
        self.k = k
        self.eigenmaps = None
        self.mask = None
        self.n_mask_voxels = None
        self.output_dir = './leica' if output_dir is None else output_dir
        self.batch_name = batch_name
        self.independent_components = None
        self.wbc = wbc

    def get_filenames(self):
        """
        use different get functions for different data sets
        create your own in utils.py
        """
        self.filenames = get_hcp_resting(self.data_dir)
        # self.filenames = your_get_function(self.data_dir)
        self.n_files = len(self.filenames)

    def full_corr_matrix(self):
        """
        Computes the group average full correlation matrix
        """
        print 'Computing full group correlation coefficient matrix ...'
        for filename in self.filenames:
            print 'loading %s...' % filename
            one_data = read_cifti_data(filename)  # for hcp cifti data
            if self.n_voxels is None:
                self.n_voxels = one_data.shape[0]
            if self.corr_matrix is None:
                self.corr_matrix = np.zeros((self.n_voxels, self.n_voxels))
            self.corr_matrix += np.corrcoef(one_data)
        self.corr_matrix /= self.n_files
        self.n_mask_voxels = self.corr_matrix.shape[0]

    def estimate_k(self, th=2):
        """
        Estimates k using z-score based method
        """
        z_matrix = scale(self.corr_matrix, axis=1)
        z_matrix[z_matrix < th] = 0
        self.k = np.count_nonzero(z_matrix) / self.n_voxels
        print 'Estimated number of nearest neighbors k is:', self.k

    def remove_poor_connected_voxels(self):
        """
        Optional step to remove poorly conencted voxels. Voxels that are not
        well connected with other voxels are considered to be not meaningful/noise.
        """
        print 'Removing poorly connected voxels ...'
        degree = np.sum(self.corr_matrix, axis=0)
        degree_ind = np.argsort(degree)
        self.mask = [degree_ind[i] for i in range(self.n_voxels / 100, self.n_voxels)]
        self.corr_matrix = self.corr_matrix[self.mask, :]
        self.corr_matrix = self.corr_matrix[:, self.mask]
        self.n_mask_voxels = self.corr_matrix.shape[0]

    def knn_corr_matrix(self):
        """
        Build a sparse k nearest neighbors correlation matrix from the full correlation matrix
        """
        print 'Computing k nearest neighbor correlation matrix ...'
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

    def laplacian_eigenmaps(self):
        """
        Computes the Laplacian Eigenmaps decomposition of the knn matrix
        """
        print 'Computing Laplacian Eigenmaps ...'
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

        eigenvals, self.eigenmaps = sparse.linalg.eigsh(self.corr_matrix, self.n_components+1, which='LA', tol=10**-10)
        self.eigenmaps = self.eigenmaps[:, -self.n_components - 1:-1]  # eliminate one evec
        print 'Eigenvalues:', 1 - eigenvals

    def ica(self):
        """
        Computes ICA and flips sign of the components
        """
        print 'Starting ICA ...'
        fica = FastICA(n_components=self.n_components, max_iter=10000)
        ic = fica.fit_transform(self.eigenmaps)
        ic = scale(ic, axis=0)
        # make components have positive peak
        for i in range(self.n_components):
            if max(ic[:, i]) + min(ic[:, i]) < 0:
                ic[:, i] = -ic[:, i]
        if self.mask is not None:
            self.independent_components = np.zeros((self.n_voxels, self.n_components))
            self.independent_components[self.mask, :] = ic
        print 'Components have shape:', self.independent_components.shape

    def save_outputs(self):
        """
        Save the ICA components, and convert both the thresholded and non-thresholded
        components to CIFTI format
        """
        print 'Saving results ...'
        if not os.path.isdir(self.output_dir):
            subprocess.call(['mkdir', '-p', output_dir])
        np.save(os.path.join(self.output_dir, self.batch_name+'_leica.npy'), self.independent_components)
        save_cifti_data(self.independent_components,
                        os.path.join(self.output_dir, self.batch_name+'_leica.dtseries.nii'), self.wbc)

        # create a temporary list that's necessary for converting CIFTI series to scalars
        list_file = open(os.path.join(self.output_dir, 'leica_list'), 'w')
        for i in range(self.n_components):
            list_file.write('%d\n' % (i+1))
        list_file.close()

        # save non-thresholded spatial maps
        subprocess.call(['wb_command', '-cifti-convert-to-scalar', os.path.join(self.output_dir,
                        self.batch_name+'_leica.dtseries.nii'), 'ROW', os.path.join(self.output_dir,
                        self.batch_name+'_leica.dscalar.nii'), '-name-file',
                        os.path.join(self.output_dir, 'leica_list')])

        # save thresholded spatial maps
        self.independent_components[self.independent_components < 2] = 0
        save_cifti_data(self.independent_components,
                        os.path.join(self.output_dir, self.batch_name + '_leica_thresholded.dtseries.nii'))

        subprocess.call(['wb_command', '-cifti-convert-to-scalar',
                         os.path.join(self.output_dir, self.batch_name + '_leica_thresholded.dtseries.nii'), 'ROW',
                         os.path.join(self.output_dir, self.batch_name + '_leica_thresholded.dscalar.nii'),
                         '-name-file', os.path.join(self.output_dir, 'leica_list')])

        subprocess.call(['rm', os.path.join(self.output_dir, 'leica_list')])

    def run_leica(self):
        """
        Run LEICA procedure
        """
        print 'Starting LEICA ...'
        self.get_filenames()
        self.full_corr_matrix()
        if self.k is None:
            self.estimate_k()
        else:
            print 'Use k = %d' % self.k
        self.remove_poor_connected_voxels()
        self.knn_corr_matrix()
        self.laplacian_eigenmaps()
        self.ica()
        self.save_outputs()

# data_dir is the root directory of the data set
# if the number of neighbors k is not specified, it will be estimated
# if the output directory is not set, the outputs will be stored in ./leica/
# wbc is the address to connectome workbench command
if __name__ == '__main__':
    data_dir = 'path_to_data/HCP_S900_100unrelated_rsfMRI_fix/'
    output_dir = './leica'
    wbc = 'path_to_workbench/workbench/bin_rh_linux64/wb_command'
    leica = LEICA(wbc, data_dir, output_dir, batch_name='test')
    leica.run_leica()


