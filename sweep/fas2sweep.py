#!/usr/bin/python
# -*- coding: utf-8 -*-
from .sweep_support import mask2vec,generate_chunk,length,ceil,size,fastaread
from .default_proj_mat_ope import check_default_proj_mat
import h5py
import os
import numpy as np
import pandas as pd
import sys
from scipy.sparse import lil_matrix,hstack
import dask.dataframe as dd
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
def fas2sweep(xfas,orth_mat=None,mask=None,verbose=False,chunk_size=1000,
              projection=True,fasta_type='AA',return_dense=None,
              n_jobs=1):
    if return_dense is None:
        if projection == True:
            return_dense = True
        else:
            return_dense = False
    if mask is None:
        mask = np.array([2,1,2])
    if fasta_type == 'AA':
        defSize = 20
    elif fasta_type == 'NT':
        defSize = 4
    mask_type = type(sum(list(mask)))
    mask_sum = sum([mask[0],mask[2]])
    if len(mask) != 3 or not (mask_type == int or mask_type == np.int32):
        message = 'Mask must be an array with 3 integer values.'
        raise Exception(message)
    elif not (orth_mat is None) and not projection:
        raise Exception('The orth_mat parameter is unnecessary if projection=False.')
    elif (mask_sum > 5 and fasta_type == 'AA') or (
            mask_sum > 10 and fasta_type == 'NT'):
        raise Exception('The size of the mask parts is too high.')
    # Extracts the sequences from the fasta file
    if isinstance(xfas, str):
        fas_cell = fastaread(xfas)
    else:
        fas_cell = xfas
    headers=[]
    seqs=[]
    for i in fas_cell:
        seqs.append(str(i.seq))
        headers.append(str(i.description))
    seqs = np.array(seqs)
    headers = np.array(headers)
    # Checking if all sequences are bigger than de mask size
    vlen = np.vectorize(len)
    seq_size = np.array(vlen(seqs))
    headers_small = seq_size < sum(mask);
    if sum(headers_small.astype(int)) > 0:
        message = 'There are sequences smaller than the mask size.'
        raise Exception(message)
    # Calculate chunks number
    chunks = ceil(len(seqs)/chunk_size)
    idx = generate_chunk(chunks, length(seqs))-1;
    rows_size = defSize**mask[0]*defSize**mask[2]
    if projection:
        if orth_mat is None:
            if rows_size != 160000:
                message = ('The default matrix is intended for the sweep of '
                           'amino acids with the default mask, for other '
                           'cases you can disable the projection or set the '
                           'orth_mat parameter.')
                raise Exception(message)
            # Download default projection matrix if not available
            libLocal = os.path.dirname(os.path.realpath(__file__))
            mat_file_local = os.path.join(libLocal,
                                          'sweep-default-projection-matrix-600.mat')
            check_default_proj_mat(mat_file_local)
            orth_mat = h5py.File(mat_file_local,'r')
            var_name = list(orth_mat.keys())[0]
            orth_mat = orth_mat[var_name][()].T
        else:
            if size(orth_mat)[0] != rows_size:
                message = ("The defined orth_mat does not have the "
                           "appropriate dimensions."
                           "\nThe number of lines must be:"
                           "\n(x**mask[0])*(x**mask[2]),"
                           "\nwhere x=20, if fasta_type==\'AA\',"
                           "\nand x=4, if fasta_type==\'NT\'.")
                raise Exception(message)
    m2v = lambda a: mask2vec(a, mask, defSize)[0]
    def sweep_chunk(i):
        parcM = seqs[np.array(range(int(idx[i,0]),int(idx[i,1])+1))]
        parcM = pd.Series(parcM)
        parcM = dd.from_pandas(parcM, npartitions=1)
        if projection:
            W160k = np.array(parcM.apply(m2v,meta=object).compute().to_list())
            W160k = np.dot(W160k,orth_mat)
        else:
            W160k = parcM.apply(m2v,meta=object).compute()
            W160k = np.array(W160k.to_list())
        return lil_matrix(W160k).T
    # Run sweep on chunks
    war='.*A worker stopped while some jobs were given to the executor\. T .*'
    warnings.filterwarnings("ignore", message=war)
    range_s=pd.Series(range(0,chunks))
    resultMat = Parallel(n_jobs=n_jobs)(delayed(sweep_chunk)(i) for i in tqdm(
        range_s,position=0,leave=True,desc='Running SWeeP',file=sys.stdout,
        disable=(not verbose)))
    resultMat=hstack(resultMat).T.tocsr()

    if return_dense:
        resultMat=resultMat.todense()
    return resultMat