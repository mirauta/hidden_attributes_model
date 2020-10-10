#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:43:06 2020

@author: mirauta
"""

from functions import *
# import seaborn as sb
# import timeit
# import sklearn.ensemble as skle
# import sklearn  as skl
# import sklearn.metrics as sklm
path="/Users/bogdanmirauta/Git/Candidaturi/CAssessment/"


df=pd.read_csv(path+'data_exam.csv',index_col=0)
df2=pd.concat([df.query('question_id==@q')[["student_id","correct"]].set_index("student_id") for q in df.question_id.unique()],1)
df2.columns=df.question_id.unique()
df=df2.astype(int)

Krange=[2,4,6,8,10,12,14,16,18,20]

proba={}
proba2={}
run_est_q=0
run_est_l=1
if run_est_q:
    print (1)
    for K in Krange:
        model=run_estimations(df[:750],tune=100,samples=10,K=K,path=path,name="uniform_learner_750_")


if run_est_l:
    for K in Krange:
        try:
            name="uniform_learner_750_"
            learn=pd.read_csv(path+name+"learner_fixed_question"+str(K)+".txt",sep="\t",index_col=0)
        except:
            try:
                question=pd.read_csv(path+name+"question_"+str(K)+".txt",sep="\t",index_col=0)
                model=run_estimations_question(df[750:].T[:49].T,question[:49],tune=100,samples=10,K=K,path=path,name="uniform_learner_750_")
            except:1
rez={}
for K in Krange:        
    try:
        name="uniform_learner_750_"
        learn=pd.read_csv(path+name+"learner_fixed_question"+str(K)+".txt",sep="\t",index_col=0)
        quest=pd.read_csv(path+name+"question_"+str(K)+".txt",sep="\t",index_col=0)
            
        # rez[K]=np.array([(pm.Bernoulli.dist(p=np.dot(learn,quest.T)).random(1)[:,49]==df[750:][49]).mean().mean() for i in np.arange(100)])
        rez[K]=pd.DataFrame(data=np.dot(learn,quest.T)[:,49],columns=['proba'],index=df[750:].index);rez[K]['real']=df[750:][49]
        conc=pd.read_csv(path+name+"concentration_"+str(K)+".txt",sep="\t",index_col=0)
            
    except:1
    
auc=plot_roc_multiple(rez)

a=np.array(list(auc.values()))

plt.figure(figsize=(5,6))
plt.subplot(2,1,1)
rez2={};rez2[16]=rez[16]
plot_roc_multiple(rez2)
plt.subplot(2,1,2)
plt.plot(np.arange(len(a)),a,'bo-',lw=2)
plt.xticks(np.arange(len(list(auc))),auc.keys())
plt.xlabel("Number of hidden attributes",fontsize=16)
plt.ylabel("Performance ( AUC)",fontsize=16)
plt.legend(loc=2)
plt.tight_layout()

plt.savefig(path+"model.png",dpi=400)
sys.exit()

model=run_estimations(df[:750],tune=100,samples=10,K=K,path=path,name="uniform_learner_750_",run=0)
model_to_graphviz(model)
