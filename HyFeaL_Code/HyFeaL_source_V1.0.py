###############################################################
# Programming instructions：
# A fast and accurate dimension reduction framework for methylation microarray data analysis using hybrid feature learning.
# This script is comprised of the functions for identifying DMS and visualization in HyFeaL computational framework.
# Final Edit Time ：2021.9.23
##############################################################

# Dependent packages
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

#X = fs.values; 
#X represents the preprocessed methylation array data, with size of n*m, n is the number of samples and m is the feature dimension;
#y = np.array(lb); 
#y represents the sample class or labels;

##############################################################
# Hybrid ensemble feature selection for identifying DMS
##############################################################
def HyFeaL_1s(X,y,Q1=0.05)):#By deualt, Q1 = 5%
    s1 = chi_square.chi_square(X, y)
    id1 = chi_square.feature_ranking(s1)[0:num_fea]
    s2 = f_score.f_score(X, y)
    id2 = f_score.feature_ranking(s2)[0:num_fea]
    s3 = fisher_score.fisher_score(X, y)
    id3 = fisher_score.feature_ranking(s3)[0:num_fea]
    s4 = reliefF.reliefF(X, y)
    id4 = reliefF.feature_ranking(s4)[0:num_fea]
    id_comb1 = list(set(id1).union(id2,id3,id4)) 
    #X_filtered = X[:,id_comb]
    return id_comb1

def Single(X,y,method,num_fea):
    if method=='chi_square':
        score = chi_square.chi_square(X, y)
        idx = chi_square.feature_ranking(s1)[0:num_fea]
    elif method=='f_score':
        s2 = f_score.f_score(X, y)
        idx = f_score.feature_ranking(s2)[0:num_fea]
    elif method=='fisher':
        s3 = fisher_score.fisher_score(X, y)
        idx = fisher_score.feature_ranking(s3)[0:num_fea]
    elif method=='reliefF':
        s4 = reliefF.reliefF(X, y)
        idx = reliefF.feature_ranking(s4)[0:num_fea]
    else:
        print('check input...')
    return idx

def HyFeaL_2s(X,y,Q2,method):
    ss = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=123)
    #By default, the number of data subsets is 30 (n_splits);
    num_fea=int(X.shape[1]*Q2)
    df = pd.DataFrame()
    for train_index, test_index in ss.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if method=='chi_square':
            idx = chi_square.feature_ranking(chi_square.chi_square(X, y))[0:num_fea]
        if method=='f_score':
            idx = f_score.feature_ranking(f_score.f_score(X, y))[0:num_fea]
        if method=='fisher':
            idx = fisher_score.feature_ranking(fisher_score.fisher_score(X, y))[0:num_fea]
        if method=='reliefF':
            idx = reliefF.feature_ranking(reliefF.reliefF(X, y))[0:num_fea]
        df = df.append(pd.Series(idx),ignore_index=True)
        df = df.astype(int)
        c = np.bincount(df.values.flat)
        d = np.where(c>=5)
        ids = np.array(d).reshape(-1,)
    return ids

def HyFeaL_3s(ids_1,ids_2,ids_3,ids_4):
    id_comb2 = list(set(ids_1).union(ids_2,ids_3,ids_4))
    return id_comb2

##############################################################
# Visualization with HyFeaL
##############################################################
from sklearn.manifold import TSNE

def Labelcorr(y):
    S = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        for j in range(i, len(y)):
            if(y[i]==y[j]):
                S[i][j] = 1
            else:
                S[i][j] = 0
            S[j][i]=S[i][j]
    return S

def SGE_tsne(X,y,perplexity):
    G1 = 1-squareform(pdist(X,metric='correlation'))
    G2 = Labelcorr(y)
    G3 = np.multiply(G1,G2)
    G3_dism = -G3+1
    em_final = TSNE(n_components=2,metric='precomputed',perplexity=perplexity).fit_transform(G3_dism) 
    return em_final
