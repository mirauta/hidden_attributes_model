#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:43:06 2020

@author: mirauta
"""

import sys

import numpy as np
import pandas as pd
from copy import deepcopy

from scipy import optimize
from scipy.cluster import hierarchy 
import sklearn.metrics as sklm
import theano
theano.config.mode = 'FAST_COMPILE' 
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

import  matplotlib.pyplot as plt 
import multiprocessing as mp
try:mp.set_start_method("fork")
except:1
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform, model_to_graphviz
import seaborn as sb

def fun():
    return
'''functions'''
def plot_roc_multiple(rezdict=None):

    roc_aucl={}
    for rezk in rezdict:
        rez=rezdict[rezk]
        fpr,tpr,a=sklm.roc_curve(rez['real'],rez['proba'])
        roc_auc = sklm.auc(fpr, tpr)
        roc_aucl[rezk]=roc_auc
        lw = 2
        plt.plot(fpr, tpr, 
                  lw=lw, label=str(rezk)+' (AUC %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    # plt.title('ROC Curve by label probability',fontsize=16)
    plt.legend(loc="lower right")
    # plt.show()
    return roc_aucl
    
def plot_roc(test=None, qfeatures=None,target=None,model=None,rez=None):
    if rez is None:
        predictions = model.predict_proba(test[qfeatures])[:,1]
        rez=pd.DataFrame(np.hstack([predictions.reshape(-1,1),test[target].values.reshape(-1,1)]),columns=['proba','real'])

    fpr,tpr,a=sklm.roc_curve(rez['real'],rez['proba'])
    roc_auc = sklm.auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.title('ROC Curve by label probability',fontsize=16)
    plt.legend(loc="lower right")
    plt.show()

def run_estimations_Bernoulli(df,    samples=10,    tune=100,Krange=np.arange(2,20),path="./",name=""):
    ch=1

    N=df.shape[0]
    Q=df.shape[1]
    for K in Krange:
        with pm.Model() as model:
            learner=pm.Bernoulli ('learner',shape=(N,K))    
            question=pm.Bernoulli ('question',a=np.repeat(.1,K),shape=(Q,K))
            x=pm.math.dot(learner, question.T)
            results= pm.Bernoulli('rezults',p=x,shape=(N,Q),observed=df)
        
        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point)  )                
        model_to_graphviz(model)
        
        with model:
            trace = pm.sample(samples,chains=ch,tune=tune, discard_tuned_samples=True)
    
    
        a=pm.math.dot(trace['learner'].mean(0), trace['question'][:,:].mean(0).T)
        pd.DataFrame(a.eval()).to_csv(path+name+"estim_"+str(K)+".txt",sep="\t")
        print ("finished: "+str(K))
    return model

def run_estimations_difficulty(df,tune=100,samples=10,Krange=np.arange(2,20),path="./",name="",run=1):
    ch=1  
    N=df.shape[0]
    Q=df.shape[1]
    for K in Krange:
        with pm.Model() as model:
            learner=pm.Uniform ('learner',shape=(N,K))    
            question=pm.Dirichlet ('question',a=np.repeat(.1,K),shape=(Q,K))
            difficulty=pm.Uniform ('difficulty',0.1,4,shape=(Q,1),testval=np.repeat(.5,Q).reshape(Q,1))
            x=pm.math.dot(learner, (difficulty*question).T)
            results= pm.Bernoulli('rezults',p=x,shape=(N,Q),observed=df)
        
        if run:
            with model:
                trace = pm.sample(samples,chains=ch,tune=tune, discard_tuned_samples=True)
    
            # a=pm.math.dot(trace['learner'].mean(0), difficulty*trace['question'][:,:].mean(0).T)
            
            pd.DataFrame(trace['learner'].mean(0)).to_csv(path+name+"learner_"+str(K)+".txt",sep="\t")
            pd.DataFrame(trace['question'].mean(0)).to_csv(path+name+"question_"+str(K)+".txt",sep="\t")
            pd.DataFrame(trace['difficulty'].mean(0)).to_csv(path+name+"difficulty_"+str(K)+".txt",sep="\t")
            # pd.DataFrame(a.eval()).to_csv(path+name+"estim_"+str(K)+".txt",sep="\t")
            print ("finished: "+str(K))
    return model

   

def run_estimations(df,tune=100,samples=10,K=2,path="./",name="",run=1):
    ch=1  
    N=df.shape[0]
    Q=df.shape[1]
    # for K in Krange:
    with pm.Model() as model:
        learner=pm.Uniform ('learner',shape=(N,K))   
        concentration=pm.Uniform('concentration',testval=.5)
        question=pm.Dirichlet ('question',a=np.repeat(concentration,K),shape=(Q,K))
        # difficulty=pm.Uniform ('difficulty',0.1,4,shape=(Q,1),testval=np.repeat(.5,Q).reshape(Q,1))
        x=pm.math.dot(learner, question.T)
        results= pm.Bernoulli('rezults',p=x,shape=(N,Q),observed=df)
    
    if run:
        with model:
            trace = pm.sample(samples,chains=ch,tune=tune, discard_tuned_samples=True)

        a=pm.math.dot(trace['learner'].mean(0), trace['question'][:,:].mean(0).T)
        
        pd.DataFrame(trace['learner'].mean(0)).to_csv(path+name+"learner_"+str(K)+".txt",sep="\t")
        pd.DataFrame(trace['question'].mean(0)).to_csv(path+name+"question_"+str(K)+".txt",sep="\t")
        # pd.DataFrame(trace['difficulty'].mean(0)).to_csv(path+name+"difficulty_"+str(K)+".txt",sep="\t")
        pd.DataFrame(a.eval()).to_csv(path+name+"estim_"+str(K)+".txt",sep="\t")
        pd.DataFrame(trace['concentration']).to_csv(path+name+"concentration_"+str(K)+".txt",sep="\t")
        print ("finished: "+str(K))
    return model


def run_estimations_question(df,question,tune=100,samples=10,K=2,path="./",name="",run=1):
    ch=1  
    N=df.shape[0]
    Q=df.shape[1]
   
    with pm.Model() as model:
        learner=pm.Uniform ('learner',shape=(N,K))    
        # question=pm.Dirichlet ('question',a=np.repeat(.1,K),shape=(Q,K))
        # difficulty=pm.Uniform ('difficulty',0.1,4,shape=(Q,1),testval=np.repeat(.5,Q).reshape(Q,1))
        x=pm.math.dot(learner, question.T)
        results= pm.Bernoulli('rezults',p=x,shape=(N,Q),observed=df)
    
    if run:
        with model:
            trace = pm.sample(samples,chains=ch,tune=tune, discard_tuned_samples=True)

        pd.DataFrame(trace['learner'].mean(0)).to_csv(path+name+"learner_fixed_question"+str(K)+".txt",sep="\t")
        # pd.DataFrame(trace['question'].mean(0)).to_csv(path+name+"question_"+str(K)+".txt",sep="\t")
        # pd.DataFrame(trace['difficulty'].mean(0)).to_csv(path+name+"difficulty_"+str(K)+".txt",sep="\t")
        # pd.DataFrame(a.eval()).to_csv(path+name+"estim_"+str(K)+".txt",sep="\t")
        print ("finished: "+str(K))
    return model

   
def run_estimations_dirichlet(df,Krange=np.arange(2,20),path="./"):
    ch=1
    samples=30
    tune=150
    N=df.shape[0]
    Q=df.shape[1]
    for K in Krange:
        with pm.Model() as model:
            learner=pm.Dirichlet ('learner',a=np.repeat(.5,K),shape=(N,K))    
            question=pm.Dirichlet ('question',a=np.repeat(.1,K),shape=(Q,K))
            x=pm.math.dot(learner, question.T)
            results= pm.Bernoulli('rezults',p=x,shape=(N,Q),observed=df)
        
        for RV in model.basic_RVs:
            print(RV.name, RV.logp(model.test_point)  )                
        model_to_graphviz(model)
        
        with model:
            trace = pm.sample(samples,chains=ch,tune=tune, discard_tuned_samples=True)
    
    
        a=pm.math.dot(trace['learner'].mean(0), trace['question'][:,:].mean(0).T)
        pd.DataFrame(a.eval()).to_csv(path+"estim_"+str(K)+".txt",sep="\t")
        print ("finished: "+str(K))
    return model
