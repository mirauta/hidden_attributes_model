#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:43:06 2020

@author: mirauta
"""

from functions import *
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-path', metavar='N', help='path to data',default=None,required=True)
parser.add_argument('-data_in', metavar='N', help='file name',default=None,required=True)
parser.add_argument('-name', metavar='N', help='name of the output',default=None,required=True)
parser.add_argument('-target', metavar='N', help='name of target field',default=None,required=True)

args = parser.parse_args()
path=args.path
data_in=args.data_in
name=args.name
target=args.target


TUNE=100
SAMPLES=10

def load_data(path,data_in,target):
    '''
    Description:
    ----------
    Function transposing input data into a matrix format
    This function needs to be updated to reflect input_data structure
    
    Parameters
    ----------
    path : string
        Path to data
    data_in : string
        name of Input data file
    target : string
        name of target field

    Returns
    -------
    The list [X_train, X_test, y_train, y_test]
    '''
        
    df=pd.read_csv(path+'data_exam.csv',index_col=0)
    df2=pd.concat([df.query('question_id==@q')[["student_id","correct"]].set_index("student_id") for q in df.question_id.unique()],1)
    df2.columns=df.question_id.unique().astype('U')
    df=df2.astype(int)
   
    X_train, X_test, y_train, y_test = train_test_split(df[np.setdiff1d(df.columns,target)], df[target], test_size=0.25, random_state=42)
  
    return [X_train, X_test, y_train, y_test]


def main_test(data,estimation_function_learn,estimation_function_test,K,estimate_parameters_flag=1, evaluate_performance_flag=1):
    '''
    Description
    ----------
    Returns a learner profile and an estimation of the prediction accuracy.
    Writes results in the PATH folder 

    Writes the parameter estimation for the generative model if estimate_parameters_flag is TRUE
    This estimation is performed for the training data
    If question parameters are already evaluated the estimate_parameters_flag shoudl be set to FALSE
    Parameters
    ----------
    data : list
    [X_train, X_test, y_train, y_test]

    estimation_function_learn : function
    
    estimation_function_test : function

    K : int
        Number of hidden factors
        
    estimate_parameters_flag : BOOL  
        If TRUE the function writes the parameter estimation for the generative model 
        This estimation is performed on and for the training data
        Default is TRUE
    evaluate_performance_flag : BOOL
        If this flag is TRUE the function evaluates the model precision using a test data set
        Requires prior evaluation of question parameters
        Compares real target to the estimation and writes a ROC plot
        Default is TRUE

    Returns
    -------
    '''   
    [X_train, X_test, y_train, y_test]=data
    df_train=pd.concat([X_train,y_train],1)
    
    try:
        model=estimation_function_learn(df_train,tune=TUNE,samples=SAMPLES,K=K,path=path,name=name,run=0)[0]
        diagr=model_to_graphviz(model)
        from graphviz import render
        diagr.render(path+name+"DAG", format="png")
        print ("Wrote succesfully the DAG")
    except: 
        print ("Cannot install render")
    
    if estimate_parameters_flag :
        
          model=estimation_function_learn(df_train,tune=TUNE,samples=SAMPLES,K=K,path=path,name=name)
    
    
    if evaluate_performance_flag :

        try:
            question=pd.read_csv(path+name+"question_"+str(K)+".txt",sep="\t",index_col=0)
        except:
            print(" Questions parameters were not estimated")
            
        model,trace=estimation_function_test(X_test,question.loc[X_test.columns],tune=TUNE,samples=10,K=K,path=path,name=name)
        learn=trace['learner'].mean(0)
        ''' 
'''
        estimation=pd.DataFrame( np.dot(learn,question.T),columns=question.index,index=X_test.index)
        rez={}
        rez[K]=pd.DataFrame(data=estimation[target].values,columns=['proba'],index=y_test.index);rez[K]['real']=y_test
                      
        plt.figure()
        auc=plot_roc_multiple(rez)
        plt.savefig(path+name+"ROC.png",dpi=100)


if __name__ == "__main__":
    data=load_data(path,data_in,target)
    main_test(data=data,estimation_function_learn=fun_infer_model_learn,estimation_function_test=fun_infer_model_test,K=3)
    