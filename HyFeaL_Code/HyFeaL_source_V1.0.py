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

#X = fs.values
#y = np.array(lb)

##############################################################
# Hybrid ensemble feature selection for identifying DMS
##############################################################
def EFS_1s(X,y,num_fea):
    s1 = chi_square.chi_square(X, y)
    id1 = chi_square.feature_ranking(s1)[0:num_fea]
    s2 = f_score.f_score(X, y)
    id2 = f_score.feature_ranking(s2)[0:num_fea]
    s3 = fisher_score.fisher_score(X, y)
    id3 = fisher_score.feature_ranking(s3)[0:num_fea]
    s4 = reliefF.reliefF(X, y)
    id4 = reliefF.feature_ranking(s4)[0:num_fea]
    id_comb = list(set(id1).union(id2,id3,id4)) 
    #X_filtered = fs[:,id_comb]
    return id_comb

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

def EFS_2s(X,y,num_fea,ms,cv):
    ss = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=123)
    df = pd.DataFrame()
    for train_index, test_index in ss.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        id_fs = Single(X_train,y_train,method=ms,num_fea)
        df = df.append(pd.Series(id_fs),ignore_index=True)
        df = df.astype(int)
        c = np.bincount(df.values.flat)
        d = np.where(c>=cv)
        e = np.array(d).reshape(-1,)
        #X_selected = fs[:,e]
    return e


##############################################################
# Visualization with graph learning and t-SNE
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

def SGE_tsne(X,y,N=25): # supervised t-SNE for low-dimensional embeddings
    G1 = 1-squareform(pdist(X,metric='correlation'))
    G2 = Labelcorr(y)
    G3 = np.multiply(G1,G2)
    G3_dism = -G3+1
    em_final = TSNE(n_components=2,metric='precomputed',perplexity=N).fit_transform(G3_dism) 
    return em_final
