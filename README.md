# HyFeaL
A fast and accurate dimension reduction framework for methylation microarray data analysis using hybrid feature learning.

## The overview of HyFeaL
![image](https://github.com/TQBio/HyFeaL/blob/main/Pictures/Fig1.png)

## Computational pipeline

### 1) The preprocess instructions of Infinium Human Methylation 450k data is available at our previous work [HyDML](https://github.com/TQBio/HyDML).

### 2) Identifying the robust DMS with HyFeaL
   
2-1: Input the preprocessed array data with size of n * m array, n is the number of the samples, m is the number of CpGs(features).
   
       Sample_id  cg_1   cg_2                cgm_m
   
       1      0.15   0.75                     0.95
    
       2      0.85   0.12                     0.25
    
       ...        ...              ...
    
       n      0.35   0.45                     0.75

2-2: Implement HyFeaL to identify the robust DMS.

       X_fs = EFS_1s(X,y,int(X.shape[1]*0.05))

