#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### NAIVE BAYES FUSION

# Inputs: ytest, ymodels should have multiple model results
# ymodels = []
# ymodels.append(ymodel_dt)
# ymodels.append(ymodel_LR)
# etc 

import numpy as np
from itertools import product

def NB_fusion(ytest, ymodels):

    class0prob = [] # label probabilities for true class 0 for each of the different classifiers
    class1prob = [] # label probabilities for true class 1 for each of the different classifiers

    prior0 = len(ytest[ytest==0]) / len(ytest) # prior probability for class 0
    prior1 = len(ytest[ytest==1]) / len(ytest) # prior probability for class 1

    for i in range(len(ymodels)):

        p0 = []
        p1 = []

        true0 = ymodels[i][np.where(ytest==0)] # samples where true value is class 0
        true0_lab0 = true0[true0==0] # true value is class 0 and label is 0
        true0_lab1 = true0[true0==1] # true value is 0 and label is 1

        p0.append(len(true0_lab0) / len(true0)) # p0[0], probability of label 0 given true 0
        p0.append(len(true0_lab1) / len(true0)) # p0[1], probablity of label 1 given true 0

        class0prob.append(p0) 

        true1 = ymodels[i][np.where(ytest==1)] # samples where true value is class 1
        true1_lab0 = true1[true1==0] # samples where true value is 1 and label is 0
        true1_lab1 = true1[true1==1] # samples where true value is 1 and label is 1

        p1.append(len(true1_lab0) / len(true1)) # p1[0], probability of label 0 given true 1
        p1.append(len(true1_lab1) / len(true1)) # p1[1], probability of label 1 given true 1

        class1prob.append(p1)

    # possible combinations for labels as an array   
    classifier_labels = np.array(list(product([0, 1], repeat=len(ymodels)))) 

    final_labels = []
    post_probs = []
    for c in classifier_labels: # for each combination of classifier labels

        post=[]
        cond0 = 1
        cond1 = 1

        for i in range(len(class0prob)):
            cond0*=class0prob[i][c[i]]

        post.append(prior0*cond0) # post[0], calculate posterior probability of class 0

        for i in range(len(class1prob)):
            cond1*=class1prob[i][c[i]]

        post.append(prior1*cond1) # post[1], calculate posterior probability of class 1

        post = np.array(post)
        post_probs.append(post) # posterior probabilities for each class for all combinations of classifier labels
        final_labels.append(post.argmax()) # final label for each combination of classifier labels

    fused_label = []  
    ymodels_t = np.transpose(np.array(ymodels))
    for i in range(len(ymodels_t)): 
        for c in range(len(classifier_labels)):
            if np.all(ymodels_t[i] == classifier_labels[c]): 
                fused_label.append(final_labels[c])

    fused_label = np.array(fused_label)

    return fused_label

