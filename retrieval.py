import numpy as np
import sklearn.datasets
import os
import itertools
import sklearn.metrics.pairwise
import pickle
import torch
import torchvision
import collections
import sklearn.metrics.pairwise
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_to_numpy = lambda t:t.detach().cpu().numpy()

from skimage import io
from PIL import Image

def get_dataset_folder_structure(train_folder,
                                test_mode = False
                                ):
    folder_structure = collections.OrderedDict({})
    classes = [d for d in sorted(os.listdir(train_folder)) if os.path.isdir(os.path.join(train_folder,d)) and d not in ['.','..']]
    if test_mode:
    #     classes = [classes[c] for c in [2,5]]
        classes = classes[:10]
    print(len(classes))
    folder_structure = collections.OrderedDict({c:[f for f in os.listdir(os.path.join(train_folder,c)) if not os.path.isdir(f)] for c in classes})
    if False:print(folder_structure)

    class_to_idx = {k:[] for k in folder_structure.keys()}
    filelist = []
    i = 0
    for ( c ,fls) in folder_structure.items():

        for fi in fls:
            filelist.append('/'.join([c,fi]))
            class_to_idx[c].append(i)
            i+=1
    return classes,class_to_idx, filelist

class Dataset(torch.utils.data.Dataset):
    def __init__(self,rootdir,filelist,transform):
        super(Dataset,self).__init__()
        self.filelist = filelist
        self.rootdir = rootdir
        self.transform = transform
    def __getitem__(self,idx):
#         import pdb;pdb.set_trace()
        fname = os.path.join(self.rootdir,
                             self.filelist[idx])
        label = self.filelist[idx].split('/')[0]
#         import pdb;pdb.set_trace()
#         pdb.set_trace()
        image = io.imread(fname)
        pil_image = Image.fromarray(image)
        
        tensor_image = self.transform(pil_image)
        return tensor_image,label,fname
        pass
    def __len__(self):
        return len(self.filelist)
        pass

    
class ClassSampler(torch.utils.data.Sampler):
    def __init__(self,
                 class_to_idx,
                 filelist,
                 ):
#         super(PKSampler,self).__init__()
        self.class_to_idx = class_to_idx
        self.classes = list(self.class_to_idx.keys())
        self.n_classes = len(self.classes)
        
        self.filelist = filelist
        pass
    def __iter__(self):
        for c in self.classes:
            files = self.class_to_idx[c]
            yield files        
        pass

    def __len__(self):
        return self.n_classes
        pass

    
import shutil,tqdm,gc

''' write all the embeddings to disk '''
def write_all_embeddings_to_disk(model,
                                 val_loader,
                                 imsize = (224,224),
                                 batch_size = 512,
                                 embed_dir='embeddings',
                                to_disk = False):
    if to_disk:
        embed_dir = 'embeddings'
        if os.path.exists(embed_dir):
            shutil.rmtree(embed_dir)
        os.makedirs(embed_dir)
        embed_info = embed_dir
    else:
        embed_info = {}
        
    classwise_numel = {}
    
    dummy = torch.ones(*((1,3) + imsize)).to(device)
    dummy_embeds = model(dummy)
    embed_len = dummy_embeds.shape[-1]
    del dummy_embeds
    with torch.no_grad():
        for i_,b in enumerate(tqdm.tqdm_notebook(val_loader)):

            bx,by,bf = b
            ci = np.unique(by)[0]
    #         print(ci,)
            classwise_numel[ci] = len(bf)
            embeds_ci = np.zeros((bx.shape[0],embed_len)).astype(np.float32) 
            n_batches = (bx.shape[0] + batch_size - 1)//batch_size

            for bi_ in range(n_batches):
                bxi = bx[bi_*batch_size : (bi_+1)*batch_size]
                bxi = bxi.to(device)
                embeds_bxi = model(bxi)
                embeds_bxi_ = tensor_to_numpy(embeds_bxi)
                embeds_ci[bi_*batch_size : (bi_+1)*batch_size,:] = embeds_bxi_
                del embeds_bxi,bxi
                gc.collect()

            filenames_ci = bf
    #         import pdb;pdb.set_trace()
            if to_disk:
            
                embeds_ci_fname = str(ci)+'_embeds.pkl'
                embeds_ci_fname = os.path.join(embed_dir,embeds_ci_fname)

                with open(embeds_ci_fname,'wb') as f:
                    pickle.dump(embeds_ci,f)
                    pickle.dump(filenames_ci,f)
        #         break
            else:
                embed_info.update({ci:(embeds_ci,filenames_ci)})
        return embed_info,classwise_numel
        
        
    
def read_embedding_from_disk(ci,embed_info):
    if embed_info.__class__ == str:
        embed_dir = embed_info
        embeds_ci_fname = str(ci)+'_embeds.pkl'
        embeds_ci_fname = os.path.join(embed_dir,embeds_ci_fname)
        with open(embeds_ci_fname,'rb') as f:
            embeds_ci = pickle.load(f)
            filenames_ci = pickle.load(f)
    elif embed_info.__class__ == dict:
        embeds_ci,filenames_ci = embed_info[ci]
         
        pass
    return embeds_ci,filenames_ci



def retrieve(embed_info,
             classes,
             query_classes = ['AJ_Cook','Aaron_Peirsol','Aaron_Sorkin'],
             nqueries_per_class = 4,
             n_retrieval = 4):
    
    import sklearn.neighbors

    query_idx_per_class = {ci:[] for ci in classes}
    all_retrieved = []
    all_retrieved_distances = []
    filenames_of_queries = []
    for c_same in query_classes:
        ci = classes.index(c_same)
        embed_i, filenames_i = read_embedding_from_disk(c_same,embed_info)
        filenames_i = np.array(filenames_i,dtype=object)

        nqueries = min(nqueries_per_class,len(filenames_i)) # the maximum number of queries will be lesses if our class doesnt have too many samples

        k = min(n_retrieval,len(filenames_i)) # the retrievals per query will be limited by the number of samples the class has

        _i = np.arange(embed_i.shape[0])
        query_idx_from_ci = np.random.choice(_i,nqueries,replace=False)
        query_idx_per_class[ci] = query_idx_from_ci

        query_embeddings = embed_i[query_idx_from_ci]
        filenames_of_queries_for_c = filenames_i[query_idx_from_ci]
        filenames_of_queries.append(filenames_of_queries_for_c)

        over_retrieved = {'idx':np.zeros((nqueries,len(classes),k)),
                          'dist':np.full((nqueries,len(classes),k),np.inf),
                          'names':np.zeros((nqueries,len(classes),k),dtype=object),} # we will retrieve k results from each class, 
                                                                                              # then compare among these n_classes*k results to again find the top k
                                                                                              # thus we over-retrieve


        ''' get the locations of the nearest k here (kd tree?) '''
        kdt = sklearn.neighbors.KDTree(embed_i,leaf_size=3) #what is leaf size for?
        dist, idx = kdt.query(query_embeddings, k=k) # we are letting the first sample be the query itself

        over_retrieved['idx'][:,ci,:],over_retrieved['dist'][:,ci,:],over_retrieved['names'][:,ci,:] = idx,dist,filenames_i[idx]


        for c_other in classes:

            if c_other == c_same:
    #             print(c_same,c_other)
                continue
            cj = classes.index(c_other)
            embed_j,filenames_j = read_embedding_from_disk(c_other,embed_info)

            filenames_j = np.array(filenames_j)
            '''
            D_ij = sklearn.metrics.pairwise.euclidean_distances(query_embeddings,
                                                         embed_j)
            '''

            ''' get the locations of nearest k here (kd tree?) '''
            kdt = sklearn.neighbors.KDTree(embed_j,leaf_size=3)
            k_for_cj = min(k,len(filenames_j))
            dist, idx = kdt.query(query_embeddings, k=k_for_cj)
            over_retrieved['idx'][:,cj,:k_for_cj],over_retrieved['dist'][:,cj,:k_for_cj],over_retrieved['names'][:,cj,:k_for_cj] = idx,dist,filenames_j[idx]
            pass


        dist_from_query = np.reshape(over_retrieved['dist'],(nqueries,-1)) # nqueries_per_class,(n_classes * k)
        nearest_k = np.argsort(dist_from_query,axis=-1)[:,:k] # pick the nearest k (note these indices are in 0 to n_classes*k, and cant directly be used to retrieve )

        reshaped_names = np.reshape(over_retrieved['names'],(nqueries,-1)) # for each query see all possible retrieval candidates 
        retrieved_for_c = [ reshaped_names[qi,nearest_k[qi]] for qi in range(nqueries)] # choose the topk for each query
        retrieved_distances_for_c= [dist_from_query[qi,nearest_k[qi]] for qi in range(nqueries)]
        all_retrieved.append(retrieved_for_c) # append to global retrieval records
        all_retrieved_distances.append(retrieved_distances_for_c)

    
    return all_retrieved,all_retrieved_distances,filenames_of_queries
    
    
    
    


def visualize_retrieval(query_classes,
                       all_retrieved,
                       all_retrieved_distances,
                        filenames_of_queries):
    
    for ci,c in enumerate(query_classes):

        retrieved_for_c = all_retrieved[ci]
        retrieved_distances_for_c = all_retrieved_distances[ci]
        n_retrieved_for_c = len(retrieved_for_c)
        names = np.concatenate((filenames_of_queries[ci][:,None],retrieved_for_c),axis=1)
        for qi_ in range(n_retrieved_for_c):

            nsubplots = retrieved_for_c[qi_].shape[0] + 1
            f,ax=plt.subplots(1,nsubplots)
            fname_qi_ = filenames_of_queries[ci][qi_].split('/')[-1]
            plt.title(fname_qi_)

            for i,imname in enumerate(names[qi_]):
                im = skimage.io.imread(imname)

                ax[i].imshow(im)
                ax[i].set_xticks([])
                ax[i].set_yticks([])

                if i>0: # the first image is the query
                    d = round(retrieved_distances_for_c[qi_][i-1],2)
                    ax[i].set_xlabel(str(d))

            plt.show()

def VAL_and_FAR(embed_info,
                classes,
                classwise_numel,
               d_range = np.linspace(0.,1.,5),
               ):
    n_classes = len(classes)
    num_total_images =sum(classwise_numel.values())
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
