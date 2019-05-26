import collections
import torch
from image_retrieval_api import read_embedding_from_disk
import sklearn.metrics.pairwise
import numpy as np


def VAL_and_FAR(embed_info,
                classes,
                classwise_numel,
               d_range = np.linspace(0.,1.,5),
               ):
    n_classes = len(classes)
    classwise_pdist =collections.OrderedDict({}) #to store distances in a class,class pair manner
    for ci,c_same in enumerate(classes):
        embed_i,filenames_i = read_embedding_from_disk(c_same,embed_info) # read the embeddings of class i from file
        for cj in range(ci,n_classes):
            c_other = classes[cj]
            embed_j,filenames_j = read_embedding_from_disk(c_other,embed_info) 
            D_ij = sklearn.metrics.pairwise.euclidean_distances(embed_i,
                                                         embed_j) # get all pairwise distances between samples of class i and samples of class j
            classwise_pdist[(ci,cj)] = D_ij 



    VAL = collections.OrderedDict({d:np.zeros((n_classes,)) for d in d_range}) # store VAL per d ,per class
    FAR = collections.OrderedDict({d:np.zeros((n_classes,)) for d in d_range}) # store FAR per d ,per class
    conf_mat = np.zeros((n_classes,n_classes)) # confusion matrix between classes 
    # import pdb;pdb.set_trace()
    for d in d_range:        
        for ij,D_ij in classwise_pdist.items(): # per class,class pair
            matched = D_ij < d # if distance < d
            if ij[0] == ij[1]:
                matched = np.triu(matched) # for class pairs like (0,0), (1,1) etc only the upper part of the matrix is unique


            i,j = ij
            conf_mat[i,j] = matched.sum() # how many samples of i and j are confusable
            conf_mat[j,i] = matched.sum() # for a symmetric confusion matrix
        np_classwise_numel = np.array(list(classwise_numel.values()))
        np_numel_outside_class = num_total_images- np_classwise_numel
        VAL[d] = np.diagonal(conf_mat)/np_classwise_numel**2 # VALrate = n_detected_same_class_samples/n_possible_same_class_samples   **( we are including identical samples)
        conf_mat_no_diag = conf_mat.copy()
        np.fill_diagonal(conf_mat_no_diag,0) 
        FAR[d] = np.sum(conf_mat_no_diag,1)/ (np_classwise_numel * np_numel_outside_class) # FARrate = n_detected_different_class_samples/n_possible_different_class_samples   
    #     import pdb;pdb.set_trace()
    return VAL,FAR,conf_mat