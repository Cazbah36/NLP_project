import sys
import cPickle as pickle
import pandas as pd
import numpy as np

from collections import defaultdict
from datetime import datetime as dt

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import metrics

class Image(object): 
    def __init__(self, X_filepath, y_filepath, col_name, testsize=0.2):
        #X_file and y_file must be in same filepath as the pipeline.
        self.filename = "/Users/danielbenton/Galvanize/capstone_ideas/"\
                        "finance/project/models"
        self.X_data = self.open_pickle(X_filepath) #DataFrame
        self.y_data = self.open_pickle(y_filepath) #DataFrame
        self.source_df = self.X_data.merge(self.y_data, how="inner", 
                                           left_index=True, right_index=True) #merged Dataframe wth nas
        
        self.column_list = list(self.X_data.columns) #list of columns
        self.column_name = col_name
        
        self.data = self.source_select(self.column_name) #DataFrame na's dropped        
        self.contents_df, self.ho_df = self.holdout() #self.data split
        
        self.holdout_y = None
        self.holdout_x = None
        self.holdout_xy()
        self.test_size = testsize
        self.xtrain = None #Dataframe of documents - na's dropped, dates match
        self.ytrain = None #Dataframe of documents - na's dropped, dates match
        self.xtest = None #Dataframe of y vals - na's dropped, dates match
        self.ytest = None #Dataframe of y vals - na's dropped, dates match
        self.pipeline_setup()
        self.t_xtrain = None #Dataframe of transformed documents == svd features
        self.t_xtest = None #Dataframe of transformed documents == svd features
        self.svd_pipeline = self.to_svd(len(self.xtrain))
        self.regularizer()
        self.svd_auc = None
        self.svc = None
        self.mlp = None
        self.gpc = None
        self.gnb = None
        self.lr = None
        self.sgd = None
        self.rndmfrst = None
        self.gboost = None
        self.params = defaultdict(dict)
        self.prediction = defaultdict(list)
        self.accuracies = defaultdict(list)
        self.recall = defaultdict(list)
        self.F1 = defaultdict(list)
        self.modeler()
        self.archive = None
        self.Kfold_crossval(n=3)
        self.predict()
        #self.predict_holdout()
        #self.class_size()

    def open_pickle(self, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def source_select(self, col_name): 
        """ Return DataFrame with only "column" of interest and target vals."""
        df = self.source_df[[col_name, self.y_data.columns[0]]]
        df = df.dropna()
        return df.reset_index(drop=True) #change this if there is messed up data

    def holdout(self):
        num_holdout = int(len(self.data) * .2)
        contents_df = self.data.iloc[:-num_holdout, :]
        ho_df = self.data.iloc[-num_holdout:, :]
        return contents_df, ho_df
    
    def holdout_xy(self): 
        col = list(self.ho_df.columns)
        y = col[-1]
        x = col[:-1]
        self.holdout_y = self.ho_df[y]
        self.holdout_x = self.ho_df[x]       
        
    def class_sizes(self, df):
        y = list(df.columns)[-1]
        class_size = df.groupby(y).count()
        print "Class balance for {} = {}".format(class_size, df)
        return class_size
        
    def pipeline_setup(self):
        xt, xT, yt, yT = train_test_split(self.contents_df.iloc[:,:-1],
                                          self.contents_df.iloc[:,-1],
                                          test_size=self.test_size, 
                                          random_state=6)

        self.xtrain, self.xtest = xt, xT
        self.ytrain, self.ytest = yt, yT
        print "pipeline setup" 
    
    def to_svd(self, n): 
        svd = Pipeline([("tfidfvec", TfidfVectorizer()),
                        ("clf", TruncatedSVD(n_components=n))])
            
        #nmf = Pipeline([("tfidfvec", TfidfVectorizer()),
        #                ('clf', NMF())])
        return svd

    def regularizer(self):
        srs_train = self.xtrain[self.column_name]
        srs_test = self.xtest[self.column_name]
        #fit svd model to max n == number of days in df
        self.svd_pipeline.fit(srs_train)
        xtrain_trans = self.svd_pipeline.transform(srs_train)
        xtest_trans = self.svd_pipeline.transform(srs_test)
        #variable selection.  Iterate through alphas and store coefs
        #dataframe of latent features by col
        trans_df = pd.DataFrame(columns = np.arange(len(xtrain_trans)))
        self.svd_auc = np.zeros(len(xtrain_trans))
        j = 0
        for i in np.linspace(1.5, .01, 101): 
            L1 = LogisticRegression(penalty="l1", C=i)
            L1.fit(xtrain_trans, self.ytrain.values)
            fpr, tpr, thresholds = metrics.roc_curve(self.ytest.values, 
                                                     L1.predict(xtest_trans))
            self.svd_auc[j] = metrics.auc(fpr, tpr)
            #preds[j] = L1.score(xtest_trans, self.ytest.values)
            trans_df = trans_df.append(pd.DataFrame(L1.coef_, index=[i]))
            j += 1
        coef_df = pd.DataFrame()
        for i in range(len(trans_df)): 
            coefs = trans_df.iloc[i][abs(trans_df.iloc[i]) > 0]
            if len(coefs) != 0:
                coef_df = coef_df.append(coefs.sort_values(ascending=False))
        coef_df = coef_df.fillna(0)  
        ax = coef_df.plot(title="Value of " + self.column_name +
                          " Coefficients as Alpha Approches Zero.", 
                          figsize=(10,10))
        fig = ax.get_figure()   
        fig.savefig("/Users/danielbenton/Galvanize/capstone_ideas/finance/"\
                    "project/data/visuals/" + self.column_name +
                    "_Coeff_alpha.png")
        self.t_xtrain = pd.DataFrame(xtrain_trans)[list(coef_df.columns)].set_index(self.ytrain.index)
        self.t_xtest = pd.DataFrame(xtest_trans)[list(coef_df.columns)].set_index(self.ytest.index)
        
        
    def modeler(self):
        # Set up Pipelines
        self.svc = SVC() 
        self.mlp = MLPClassifier()
        self.gpc = GaussianProcessClassifier()
        self.gnb = GaussianNB()
        self.lr = LogisticRegression(fit_intercept=False, intercept_scaling=0)
        self.sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                 random_state=42, max_iter=5, tol=None)
        self.rndmfrst = RandomForestClassifier(n_jobs=-1, oob_score=True)
        self.gboost = GradientBoostingClassifier()

    def Kfold_crossval(self, n=3): 

        Kfold = 0
        #Kfold Cross Validate
        kf = KFold(n_splits=n)
        for train_index, test_index in kf.split(self.t_xtrain):   

            X_train = self.t_xtrain.values[train_index]
            X_test = self.t_xtrain.values[test_index]
            y_train = self.ytrain.values[train_index]
            y_test = self.ytrain.values[test_index]

            Kfold +=1    
            self.svc.fit(X_train, y_train)
            svc_acc = np.mean(self.svc.predict(X_test) == y_test)
            self.accuracies["svc"].extend([svc_acc])

            self.mlp.fit(X_train, y_train)
            mlp_acc = np.mean(self.mlp.predict(X_test) == y_test)
            self.accuracies["mlp"].extend([mlp_acc])

            self.gpc.fit(X_train, y_train)
            gpc_acc = np.mean(self.gpc.predict(X_test) == y_test)
            self.accuracies["gpc"].extend([gpc_acc])

            self.gnb.fit(X_train, y_train)
            gnb_acc = np.mean(self.gnb.predict(X_test) == y_test)
            self.accuracies["gnb"].extend([gnb_acc])

            self.lr.fit(X_train, y_train)
            lr_acc = np.mean(self.lr.predict(X_test) == y_test)
            self.accuracies["lr"].extend([lr_acc])

            self.sgd.fit(X_train, y_train)
            sgd_acc = np.mean(self.sgd.predict(X_test) == y_test)
            self.accuracies["sgd"].extend([sgd_acc])
        
            self.rndmfrst.fit(X_train, y_train)
            rndmfrst_acc = np.mean(self.rndmfrst.predict(X_test) == y_test)
            self.accuracies["rndmfrst"].extend([rndmfrst_acc])
            
            self.gboost.fit(X_train, y_train)
            gboost_acc = np.mean(self.gboost.predict(X_test) == y_test)
            self.accuracies["gboost"].extend([gboost_acc])
            
            self.archives = self.accuracies.copy()
            
        print "Mean Accuracy of train_svc ", np.mean(self.accuracies["svc"])
        print "Mean Accuracy of train_mlp ", np.mean(self.accuracies["mlp"])
        print "Mean Accuracy of train_gpc ", np.mean(self.accuracies["gpc"])
        print "Mean Accuracy of train_gnb ", np.mean(self.accuracies["gnb"])
        print "Mean Accuracy of train_lr ", np.mean(self.accuracies["lr"])
        print "Mean Accuracy of train_sgd ", np.mean(self.accuracies["sgd"])
        print "Mean Accuracy of train_rndmfrst ", np.mean(self.accuracies["rndmfrst"])
        print "Mean Accuracy of train_gboost ", np.mean(self.accuracies["gboost"])
        
    def accuracy_clense(self): 
        for key in self.accuracies:
            self.accuracies[key] = []
        
    def predict(self):
        self.accuracies["test_svc"] = \
            np.mean(self.svc.predict(self.t_xtest) == self.ytest)

        self.accuracies["test_mlp"] = \
            np.mean(self.mlp.predict(self.t_xtest) == self.ytest)

        self.accuracies["test_gpc"] = \
            np.mean(self.gpc.predict(self.t_xtest) == self.ytest)

        self.accuracies["test_gnb"] = \
            np.mean(self.gnb.predict(self.t_xtest) == self.ytest)

        self.accuracies["test_lr"] = \
            np.mean(self.lr.predict(self.t_xtest) == self.ytest)
        
        self.accuracies["test_sgd"] = \
            np.mean(self.sgd.predict(self.t_xtest) == self.ytest)
        
        self.accuracies["test_rndmfrst"] = \
            np.mean(self.rndmfrst.predict(self.t_xtest) == self.ytest)
        
        self.accuracies["test_gboost"] = \
            np.mean(self.gboost.predict(self.t_xtest) == self.ytest)
        
        print "test_svc accuracies: ", self.accuracies["test_svc"]
        print "test_mlp accuracies: ", self.accuracies["test_mlp"]
        print "test_gpc accuracies: ", self.accuracies["test_gpc"]
        print "test_gnb accuracies: ", self.accuracies["test_gnb"]
        print "test_lr accuracies: ", self.accuracies["test_lr"]
        print "test_sgd accuracies: ", self.accuracies["test_sgd"]
        print "test_rndmfrst accuracies: ", self.accuracies["test_rndmfrst"]
        print "test_gboost accuracies: ", self.accuracies["test_gboost"]
        
    def predict_holdout(self):
        self.accuracies["holdout_svc"] = \
            np.mean(self.svc.predict(self.holdout_x) == self.holdout_y)
            
        self.accuracies["holdout_mlp"] = \
            np.mean(self.mlp.predict(self.holdout_x) == self.holdout_y)
            
        self.accuracies["holdout_gpc"] = \
            np.mean(self.gpc.predict(self.holdout_x) == self.holdout_y)
            
        self.accuracies["holdout_gnb"] = \
            np.mean(self.gnb.predict(self.holdout_x) == self.holdout_y)
            
        self.accuracies["holdout_lr"] = \
            np.mean(self.lr.predict(self.holdout_x) == self.holdout_y)
        
        self.accuracies["holdout_sgd"] = \
            np.mean(self.sgd.predict(self.holdout_x) == self.holdout_y)
        
        self.accuracies["holdout_rndmfrst"] = \
            np.mean(self.rndmfrst.predict(self.holdout_x) == self.holdout_y)
        
        self.accuracies["holdout_gboost"] = \
            np.mean(self.gboost.predict(self.holdout_x) == self.holdout_y)

        print "holdout_svc accuracies ", self.accuracies["holdout_svc"]
        print "holdout_mlp accuracies ", self.accuracies["holdout_mlp"]
        print "holdout_gpc accuracies ", self.accuracies["holdout_gpc"]
        print "holdout_gnb accuracies ", self.accuracies["holdout_gnb"]
        print "holdout_lr accuracies: ", self.accuracies["holdout_lr"]
        print "holdout_sgd accuracies: ", self.accuracies["holdout_sgd"]
        print "holdout_rndmfrst accuracies: ", self.accuracies["holdout_rndmfrst"]
        print "holdout_gboost accuracies: ", self.accuracies["holdout_gboost"]
