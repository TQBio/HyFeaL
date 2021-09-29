# HyFeaL
A fast and accurate dimension reduction framework for methylation microarray data analysis using hybrid feature learning.

## The overview of HyFeaL
![image](https://github.com/TQBio/HyFeaL/blob/main/Pictures/Fig1.png)

## Computational pipeline

### 1) The preprocess instructions of Infinium Human Methylation 450k data is available at our previous work [HyDML](https://github.com/TQBio/HyDML).

### 2) Identifying the robust DMS with HyFeaL
   
2-1: Input the preprocessed array data with size of n * m array, n is the number of the samples, m is the number of CpGs(features).
   
       Sample_id  cg_1   cg_2                cg_m
   
       1      0.15   0.75                     0.95
    
       2      0.85   0.12                     0.25
    
       ...        ...              ...
    
       n      0.35   0.45                     0.75

2-2: Implement HyFeaL to identify the robust DMS.

       id_comb1 = HyFeaL_1s(X,y,Q1=0.05)
       
       id_s1 = HyFeaL_2s(X,y,Q2=0.2,method='chi_square')
       
       id_s2 = HyFeaL_2s(X,y,Q2=0.2,method='fisher')
       
       id_s3 = HyFeaL_2s(X,y,Q2=0.2,method='f_score')
       
       id_s4 = HyFeaL_2s(X,y,Q2=0.2,method='reliefF')
       
       id_comb2 = HyFeaL_3s(id_s1,id_s2,id_s3,id_s4)
       
       X_fs = X[:,id_comb2]
       
 2-3: Implement HyFeaL for visualization.
 
       X_2d = SGE_tsne(X_fs,y,perplexity=25)
       
       
       
       

