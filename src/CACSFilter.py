# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:00:00 2020

@author: Bernhard Foellmer

"""

import pandas as pd
pd.options.mode.chained_assignment = None
from inspect import isfunction
import numpy as np
import math
from collections import defaultdict

from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from graphviz import Source
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy.random import shuffle

def autosize_excel_columns(worksheet, df):
    """ autosize_excel_columns
    """
    
    autosize_excel_columns_df(worksheet, df.index.to_frame())
    autosize_excel_columns_df(worksheet, df, offset=df.index.nlevels)

def autosize_excel_columns_df(worksheet, df, offset=0):
    """ autosize_excel_columns_df
    """
    
    for idx, col in enumerate(df):
        series = df[col]
        max_len = max((  series.astype(str).map(len).max(),      len(str(series.name)) )) + 1        
        max_len = min(max_len, 100)
        worksheet.set_column(idx+offset, idx+offset, max_len)
    
def df_to_excel(writer, sheetname, df):
    """ df_to_excel
    """
    
    df.to_excel(writer, sheet_name = sheetname, freeze_panes = (df.columns.nlevels, df.index.nlevels))
    autosize_excel_columns(writer.sheets[sheetname], df)
    
def format_differences(worksheet, levels, df_ref, df_bool, format_highlighted):
    """ format_differences
    """
    
    for i in range(df_ref.shape[0]):
        for j in range(df_ref.shape[1]):
            if df_bool.iloc[i,j]:
                v  = df_ref.iloc[i,j]
                try:
                    if v!=v:
                        worksheet.write_blank(i+1, j+levels, None, format_highlighted)
                    else:                                        
                        worksheet.write(i+1, j+levels, v, format_highlighted)
                except:
                    print("An exception occurred "+type(v))                    
    return

class DISCHARGEFilter:
    """ Create DISCHARGEFilter
    Filer DISCHARGE column and return boolean pandas series
        
    """
    def __init__(self):
        """ Init DISCHARGEFilter
        """
        self.filtetype = 'FILTER'
        self.feature = ''
        self.minValue = ''
        self.maxValue = ''
        self.exactValue = ''
        self.filter0 = ''
        self.filter1 = ''
        self.mapFunc = ''
        self.color = ''
        self.df_filt=None
        self.updateCACS=True
        
    def createFilter(self, feature='', minValue=None, maxValue=None, exactValue=[], mapFunc=None, name='', color='', updateCACS=True):
        """ Create normal filter
        
        :param feature: Name of the column which is filtered
        :type feature: str
        :param minValue: Minimum value if filterd by minimal boundary else None
        :type minValue: float
        :param maxValue: Maximum value if filterd by maximum boundary else None
        :type maxValue: float
        :param exactValue: List of values, if element of column is equal to element in the list, True is returnd for element in the column 
        :type exactValue: list
        :param mapFunc: Function which is applied to an element of the column before compared
        :type mapFunc: function
        :param name: Name of the boolean column whcih is added to the output file
        :type name: str
        """
        
        self.filtetype = 'FILTER'
        self.feature = feature
        self.minValue = minValue
        self.maxValue = maxValue
        self.exactValue = exactValue
        self.mapFunc = mapFunc
        self.color = color
        self.updateCACS = updateCACS
        if name:
            self.name = name
        else:
            self.name = feature
    
    def createFilterJoint(self, filter0, filter1, filtetype='AND', name='FilterJoint', updateCACS=True):
        """ Create joint filter consisting of two normal filter
        
        :param name: Name of the boolean column whcih is added to the output file
        :type name: str
        :param filter0: Filter which is combined with filter1 and operation from type filtetype
        :type filter0: DISCHARGEFilter
        :param filter1: Filter which is combined with filter0 and operation from type filtetype
        :type filter1: DISCHARGEFilter
        :param filtetype: Name of the combination operation of the two filters filter0 and filter1 (e.g. AND, OR)
        :type filtetype: str
        """
        
        self.filtetype = filtetype
        self.filter0 = filter0
        self.filter1 = filter1
        self.name = name
        self.updateCACS = updateCACS
        
    def filter(self, df):
        """ Filter dataframe df
        """
        
        if self.filtetype=='FILTER':
            NotNAN = ~df[self.feature].isna()
            if isfunction(self.mapFunc):
                df_filt = df[self.feature].apply(self.mapFunc)
                return df_filt
            else:
                df_feature = df[self.feature]
            if self.exactValue:
                df_filt = pd.Series(data=False, index=df_feature.index)
                for v in self.exactValue:
                    df_filt = df_filt | (df_feature == v)
                df_filt = df_filt & NotNAN
                return df_filt
            if self.minValue is not None and self.maxValue is None:
                df_filt = df_feature >= self.minValue
                df_filt = df_filt & NotNAN
                return df_filt
            if self.minValue is None and self.maxValue is not None:
                df_filt = df_feature <= self.maxValue
                df_filt = df_filt & NotNAN
                return df_filt
            if  self.minValue is not None and self.maxValue is not None:
                df_filt = (df_feature >= self.minValue) & (df_feature <= self.maxValue)
                df_filt = df_filt & NotNAN
                return df_filt
            
        elif self.filtetype=='AND':
            df_filt0 = self.filter0.filter(df)
            df_filt1 = self.filter1.filter(df)
            df_filt = df_filt0 & df_filt1
            return df_filt
        
        elif self.filtetype=='OR':
            df_filt0 = self.filter0.filter(df)
            df_filt1 = self.filter1.filter(df)
            df_filt = df_filt0 | df_filt1
            return df_filt
        else:
            raise ValueError('Filtetype: ' + self.filtetype + ' does not exist.')
        return None
    
    def __str__(self):
        """ print filer (e.g. print(filter))
        """
        out = ['Filter:']
        out.append('Feature: ' + self.feature)
        out.append('minValue: ' + str(self.minValue))
        out.append('maxValue: ' + str(self.maxValue))
        out.append('exactValue: ' + str(self.exactValue))
        return '\n'.join(out)
    
    

def confidencePredictor(df_in, label = 'CACS'):
    
    df = df_in.copy()
    sheet = 'linear'
    
    # Define features
    features=[]
    
    # ReconstructionDiameter
    features.append(defaultdict(lambda: None, {'FEATURE': 'ReconstructionDiameter_01', 'COLUMNE': 'ReconstructionDiameter', 'FUNCTION':  lambda v : v}))
    
    # Count
    features.append(defaultdict(lambda: None, {'FEATURE': 'Count_01', 'COLUMNE': 'Count', 'FUNCTION':  lambda v : float(v)}))
    
    # SliceThickness 3.0
    features.append(defaultdict(lambda: None, {'FEATURE': 'SliceThickness_30', 'COLUMNE': 'SliceThickness', 'FUNCTION':  lambda v : float(v==3.0)}))
    
    # SliceThickness 2.5
    features.append(defaultdict(lambda: None, {'FEATURE': 'SliceThickness_25', 'COLUMNE': 'SliceThickness', 'FUNCTION':  lambda v : float(v==2.5)}))
    
    # Modality
    features.append(defaultdict(lambda: None, {'FEATURE': 'Modality_01', 'COLUMNE': 'Modality', 'FUNCTION':  lambda v : round(v=='CT')}))
    
    # Site
    features.append(defaultdict(lambda: None, {'FEATURE': 'Site_01', 'COLUMNE': 'Site', 'FUNCTION':  lambda v : round(v in ['P10', 'P13', 'P29'])}))

    # ProtocolName includes 'cascore'
    def map_ProtocolName_CaScore(v):
        if type(v) == str:
            return float('cascore' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComments_CaScore', 'COLUMNE': 'ProtocolName', 'FUNCTION':  map_ProtocolName_CaScore}))
    
    # ProtocolName includes 'cta'
    def map_ProtocolName_CTA(v):
        if type(v) == str:
            return float('cta' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComments_cta', 'COLUMNE': 'ProtocolName', 'FUNCTION':  map_ProtocolName_CTA}))
    
    # ImageComments includes 'calcium'
    def map_ImageComment_Calcium(v):
        if type(v) == str:
            return float('calcium' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComment_calcium', 'COLUMNE': 'ImageComments', 'FUNCTION':  map_ImageComment_Calcium}))
    
    # ImageComments includes 'calcium'
    def map_ImageComment_CTA(v):
        if type(v) == str:
            return float('cta' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComment_cta', 'COLUMNE': 'ImageComments', 'FUNCTION':  map_ImageComment_CTA}))
    
    # SeriesDescription includes 'cascore'
    def map_SeriesDescription_cascore(v):
        if type(v) == str:
            return float('cascore' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'SeriesDescription_CaScore', 'COLUMNE': 'SeriesDescription', 'FUNCTION':  map_SeriesDescription_cascore}))

    # Extract features
    df_features = pd.DataFrame()
    for f in features:
        df_f = df[f['COLUMNE']].apply(f['FUNCTION'])
        df_f = df_f.rename(f['FEATURE'])
        df_features = pd.concat([df_features, df_f], axis=1)
        
    # Read label
    X = np.array(df_features)
    Y = np.array(df[label])
    
    # Replace nan by -1
    X = np.nan_to_num(X, nan=-1)
    
    # Split data in train and test
    indices = np.arange(X.shape[0])
    shuffle(indices)
    indices_train, indices_test = train_test_split(indices, test_size=0.3)
    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = Y[indices_train]
    y_test = Y[indices_test]
    
    # Train random forest
    clfRF = RandomForestClassifier(max_depth=10, n_estimators=100)
    clfRF.fit(X_train, y_train)
    
    # Extract confusion matrix and accuracy
    pred_test = clfRF.predict(X_test)
    C = confusion_matrix(pred_test, y_test)
    ACC = accuracy_score(pred_test, y_test)

    # Predict confidence
    prop = clfRF.predict_proba(X)
    confidence = (np.max(prop, axis=1)-0.5)*2
    confidence_idx = np.argsort(confidence)
    confidence_sort = confidence[confidence_idx]
    
    # Create confidence dataframe
    #df_confidence = df + pd.DataFrame(confidence, columns=['Confidence'])
    df_confidence = pd.DataFrame(confidence, columns=['Confidence'])
    
    return df_confidence, C, ACC
    
def discharge_cluster(filepath_discharge, filepath_discharge_filt):
    
    
    #filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_16042020.xlsx'
    #filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_28022020.xlsx'
    filepath_discharge_filt = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags.xlsx'  
    
    # Read discharge tags from linear sheet
    print('Reading file', filepath_discharge)
    sheet = 'linear'
    df = pd.read_excel(filepath_discharge_filt, sheet) 

    # Define features
    features=[]
    
    # ReconstructionDiameter
    features.append(defaultdict(lambda: None, {'FEATURE': 'ReconstructionDiameter_01', 'COLUMNE': 'ReconstructionDiameter', 'FUNCTION':  lambda v : v}))
    
    # Count
    features.append(defaultdict(lambda: None, {'FEATURE': 'Count_01', 'COLUMNE': 'Count', 'FUNCTION':  lambda v : float(v)}))
    
    # SliceThickness 3.0
    features.append(defaultdict(lambda: None, {'FEATURE': 'SliceThickness_30', 'COLUMNE': 'SliceThickness', 'FUNCTION':  lambda v : float(v==3.0)}))
    
    # SliceThickness 2.5
    features.append(defaultdict(lambda: None, {'FEATURE': 'SliceThickness_25', 'COLUMNE': 'SliceThickness', 'FUNCTION':  lambda v : float(v==2.5)}))
    
    # Modality
    features.append(defaultdict(lambda: None, {'FEATURE': 'Modality_01', 'COLUMNE': 'Modality', 'FUNCTION':  lambda v : round(v=='CT')}))
    
    # Site
    features.append(defaultdict(lambda: None, {'FEATURE': 'Site_01', 'COLUMNE': 'Site', 'FUNCTION':  lambda v : round(v in ['P10', 'P13', 'P29'])}))

    # ProtocolName includes 'cascore'
    def map_ProtocolName_CaScore(v):
        if type(v) == str:
            return float('cascore' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComments_CaScore', 'COLUMNE': 'ProtocolName', 'FUNCTION':  map_ProtocolName_CaScore}))
    
    # ProtocolName includes 'cta'
    def map_ProtocolName_CTA(v):
        if type(v) == str:
            return float('cta' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComments_cta', 'COLUMNE': 'ProtocolName', 'FUNCTION':  map_ProtocolName_CTA}))
    
    # ImageComments includes 'calcium'
    def map_ImageComment_Calcium(v):
        if type(v) == str:
            return float('calcium' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComment_calcium', 'COLUMNE': 'ImageComments', 'FUNCTION':  map_ImageComment_Calcium}))
    
    # ImageComments includes 'calcium'
    def map_ImageComment_CTA(v):
        if type(v) == str:
            return float('cta' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'ImageComment_cta', 'COLUMNE': 'ImageComments', 'FUNCTION':  map_ImageComment_CTA}))
    
    # SeriesDescription includes 'cascore'
    def map_SeriesDescription_cascore(v):
        if type(v) == str:
            return float('cascore' in v.lower())
        else:
            return 0
    features.append(defaultdict(lambda: None, {'FEATURE': 'SeriesDescription_CaScore', 'COLUMNE': 'SeriesDescription', 'FUNCTION':  map_SeriesDescription_cascore}))

    # Extract features
    df_features = pd.DataFrame()
    for f in features:
        df_f = df[f['COLUMNE']].apply(f['FUNCTION'])
        df_f = df_f.rename(f['FEATURE'])
        df_features = pd.concat([df_features, df_f], axis=1)
        
    # Read CACS
    #df_CACS_raw = pd.read_excel(filepath_discharge_filt, sheet)
    #df_CACS = round(df_CACS_raw['CACS'])
    df_CACS = round( df['CACS'])
    Y_CACS = np.array(df_CACS)
    
    # Nomalize features
    X = np.array(df_features)
    X1 = np.nan_to_num(X, copy=True, nan=0)
    Xz = zscore(X1)
    
    # Create feature dataframe with CACS column
    df_X = df_features.copy()
    df_X['CACS'] = df_CACS
    df_X = df_X.rename(columns={0: 'CACS'})
    
    # GaussianNB
    clfGNB = GaussianNB()
    clfGNB.fit(Xz, Y_CACS)
    pred_GNB = clfGNB.predict(Xz)
    MGNB = confusion_matrix(pred_GNB, Y_CACS)
    X_GNB = np.column_stack([idx_false, X1[idx_false], Y_CACS[idx_false], pred_one[idx_false]])
    columns = list(df_features.columns)
    colM = list()
    colM = colM + ['idx']
    colM = colM + columns
    colM = colM + ['Y_CACS']
    colM = colM + ['pred_GNB']
    df_GNB = pd.DataFrame(X_GNB, columns=colM)
    
    # One-Class-SVM
    XzOne = Xz[Y_CACS==1]
    #clfOne = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    SVMOne = svm.OneClassSVM(kernel="rbf", gamma=0.1)
    SVMOne.fit(XzOne)
    pred_one = SVMOne.predict(Xz)
    pred_one[pred_one==-1] = 0
    MOne = confusion_matrix(pred_one, Y_CACS)
    idx_false = np.where(pred_one != Y_CACS)[0]
       
    # Show comparison dataframe
    columns = list(df_features.columns)
    colM = list()
    colM = colM + ['idx']
    colM = colM + columns
    colM = colM + ['Y_CACS']
    colM = colM + ['pred_one']
    X_M = np.column_stack([idx_false, X1[idx_false], Y_CACS[idx_false], pred_one[idx_false]])
    df_M = pd.DataFrame(X_M, columns=colM)
    
    # kmeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(Xz)
    pred_kmeans = kmeans.labels_   
    MKmeans = confusion_matrix(pred_kmeans, Y_CACS)
    idx_false = np.where(pred_kmeans != Y_CACS)[0]
    
    # Show comparison dataframe
    columns = list(df_features.columns)
    colM = list()
    colM = colM + ['idx']
    colM = colM + columns
    colM = colM + ['Y_CACS']
    colM = colM + ['pred_one']
    X_M = np.column_stack([idx_false, X1[idx_false], Y_CACS[idx_false], pred_kmeans[idx_false]])
    df_M = pd.DataFrame(X_M, columns=colM)
    
    # TSNE
    X_embedded = TSNE(n_components=2).fit_transform(Xz)
    #X_embedded = PCA(n_components=2).fit_transform(Xz)

    # plt.figure()
    # colors = 'r', 'g'
    #Y_CACSp=Y_CACS[0:1000]
    #plt.scatter(X_embedded[Y_CACSp==1,0], X_embedded[Y_CACSp==1,1], colors[0], label=['1'])
    #plt.scatter(X_embedded[:,0], X_embedded[:,1], colors ,label=['0', '1'])
    #plt.legend()
    #plt.show()    
    
    # plt.figure(figsize=(6, 5))
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    # for i, c, label in zip(target_ids, colors, digits.target_names):
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    # plt.legend()
    # plt.show()
    
    # N = 50
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    # colors = np.random.rand(N)
    # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    
    plt.scatter(X_embedded[Y_CACS==1,0], X_embedded[Y_CACS==1,1], c='b')
    #plt.scatter(X_embedded[Y_CACS==0,0], X_embedded[Y_CACS==0,1], c='g')
    plt.scatter(X_embedded[idx_false,0], X_embedded[idx_false,1], c='r')
    #plt.axis([-60,-40, -14, -10])
    plt.show()
    
    colW = list()
    col = list(df_features.columns)
    colW = colW + ['idx']
    colW = colW + col
    colW = colW + ['Y_CACS']
    colW = colW + ['pred_one']
    X_df = pd.DataFrame(w, columns=colW)
    
    #kmeans = KMeans(n_clusters=2, random_state=0).fit(Xz)
    #labels = kmeans.labels_    
    
    # Decision tree
    treeClass = DecisionTreeClassifier(max_depth=5)
    treeClass.fit(X1, Y_CACS)
    pred_tree = treeClass.predict(X1)
    
    graph = Source( tree.export_graphviz(treeClass, out_file=None, feature_names=df_features.columns))
    graph.format = 'png'
    graph.render('H:/cloud/cloud_data/Projects/CACSFilter/src/tmp/graph.png',view=True)
    
    Mtree = confusion_matrix(pred_tree, Y_CACS)
    idx_false = np.where(pred_tree != Y_CACS)[0]
    
    columns = list(df_features.columns)
    colM = list()
    colM = colM + ['idx']
    colM = colM + columns
    colM = colM + ['Y_CACS']
    colM = colM + ['pred_tree']
    X_M = np.column_stack([idx_false, X1[idx_false], Y_CACS[idx_false], pred_tree[idx_false]])
    df_M = pd.DataFrame(X_M, columns=colM) 
    
    
    p = treeClass.predict_proba(X1)
    
    ###############################################
    MtestList=[]
    AccList=[]
    idx_falseList=[]
    for d in range(1,30):
        X_train, X_test, y_train, y_test, idx_train, idx2_test = train_test_split(X1, Y_CACS, test_size=0.3)
        treeClass = DecisionTreeClassifier(max_depth=10)
        treeClass.fit(X_train, y_train)
        pred_test = treeClass.predict(X_test)
        Mtest = confusion_matrix(pred_test, y_test)
        MtestList.append(Mtest)
        acc = accuracy_score(pred_test, y_test)
        AccList.append(acc)
        idx_false = np.where(pred_test != y_test)[0]
        idx_falseList.append(idx_false)
        
    # Random forest
    indices = np.arange(X1.shape[0])
    shuffle(indices)
    indices_train, indices_test = train_test_split(indices, test_size=0.3)
    X_train = X1[indices_train]
    X_test = X1[indices_test]
    y_train = Y_CACS[indices_train]
    y_test = Y_CACS[indices_test]
    
    clfRF = RandomForestClassifier(max_depth=10, n_estimators=100)
    clfRF.fit(X_train, y_train)
    pred_test = clfRF.predict(X_test)
    Mtest = confusion_matrix(pred_test, y_test)
    MtestList.append(Mtest)
    p = clfRF.predict_proba(X_test)
    p_max = np.max(p,axis=1)
    conf = (p_max-0.5)*2
    idx = np.argsort(conf)
    conf_sort = conf[idx]
    
    idx_conf = indices_test[idx]
    
    
    
    # One-Class-SVM
    XzOne = Xz[Y_CACS==1]
    #clfOne = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clfOne = svm.OneClassSVM(kernel="rbf", gamma=0.1)
    clfOne.fit(XzOne)
    pred_one = clfOne.predict(Xz)
    pred_one[pred_one==-1] = 0
    
    confusion_matrix(pred_one, Y_CACS)
    
    MOne = confusion_matrix(pred_one, Y_CACS)
    idx = np.where(pred_one != Y_CACS)[0]
    
    v = df.loc[idx[0]] 
    v1 = df_features.loc[idx]
    
    w = np.column_stack([idx, X1[idx], Y_CACS[idx], pred_one[idx]])
    
    
    q1=df_features.loc[117]
    q2=df_features.loc[137]
    
    
    # Create dataframe with false samples
    colW = list()
    col = list(df_features.columns)
    colW = colW + ['idx']
    colW = colW + col
    colW = colW + ['Y_CACS']
    colW = colW + ['pred_one']
    X_df = pd.DataFrame(w, columns=colW)
    


    
    
def discharge_filter(filepath_discharge, filepath_discharge_filt, discharge_filter_list=[]):
    """ Filter DISCHARGE excel sheet
    
    :param filepath_discharge: Filpath of the input DISCHARGE excel sheet
    :type filepath_discharge: str
    :param filepath_discharge_filt: Filpath of the output DISCHARGE excel sheet with boolean columns from filters
    :type filepath_discharge_filt: str
    :param discharge_filter_list: List of DISCHARGEFilter
    :type discharge_filter_list: list
    """
    
    # Read discharge tags from linear sheet
    print('Reading file', filepath_discharge)
    sheet = 'linear'
    df_org = pd.read_excel(filepath_discharge, sheet) 
    df = df_org.copy()
    
    # Drop unnamed
    df.drop('Unnamed: 0',axis=1, inplace=True)
    
    # Drop columns
    df.drop(columns=['InstanceNumber', 'AcquisitionTime', 'NumberOfFrames'], inplace=True)
    
    # Iterate over filters
    df_CACS = pd.DataFrame(data=True, index=df_org.index, columns = ['CACS'])
    for filt in discharge_filter_list:
        print('Apply filter:', filt.name)
        # Filtr dataframe
        df_filt = filt.filter(df)
        filt.df_filt = df_filt
        
        # Append filter column
        df[filt.name + '_OK'] = df_filt
        
        # Update df_CACS
        if filt.updateCACS:
            df_CACS['CACS'] = df_CACS['CACS'] & df_filt
        
        
    # Compute CACS column
    df_linear = df.copy()
    df_linear['CACS'] = df_CACS['CACS']
    
    # Compute confidence
    df_confidence,  C, ACC = confidencePredictor(df_linear, label = 'CACS')
    print('Accuracy:', ACC)
    print('Confusion matrix:', C)
    df_linear = pd.concat([df_linear,df_confidence], axis=1)
    
    # Create ordered list
    df_ordered = df_linear.set_index(['PatientID','StudyInstanceUID','SeriesInstanceUID'])
    df_ordered.sort_index(inplace=True)
    
    # Create workbook list    
    print('Create workbook')
    writer = pd.ExcelWriter(filepath_discharge_filt)            
    df_to_excel(writer, "ordered", df_ordered)    
    df_to_excel(writer, "linear", df_linear)    
    workbook  = writer.book
    
    print('Highlight CACS')
    
    # Highlight CACS
    format_red = workbook.add_format({'font_color': 'red'})
    for i in range(df_ordered['CACS'].shape[0]):
        if df_ordered['CACS'][i]:
            first_row=i+1
            last_row=i+1
            first_col=0
            last_col=1000
            #print(i, first_row, last_row, first_col, last_col)
            print('Highlight row:', i)
            #sys.exit()
            writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'no_blanks','format': format_red})
            writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'blanks','format': format_red})

    # Highlight features
    for filt in discharge_filter_list:
        if filt.color:
            format_red = workbook.add_format({'font_color': filt.color})
            for i in range(filt.df_filt.shape[0]):
                if filt.df_filt[i]:
                    first_row=i+1
                    last_row=i+1
                    first_col=0
                    last_col=1000
                    writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'no_blanks','format': format_red})
                    writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'blanks','format': format_red})
    
    
    writer.save()
    return 

if __name__=='__main__':
    
    # Set parameter
    filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_16042020.xlsx'
    #filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_28022020.xlsx'
    filepath_discharge_filt = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags.xlsx'  
    

    # Create ReconstructionDiameter filter
    ReconstructionDiameterFilter = DISCHARGEFilter()
    ReconstructionDiameterFilter.createFilter(feature='ReconstructionDiameter', maxValue=220, updateCACS=True)
    
    # Create SliceThickness filter for 3.0 mm
    SliceThickness30Filter = DISCHARGEFilter()
    SliceThickness30Filter.createFilter(feature='SliceThickness', exactValue=[3.0], name='SliceThickness30', updateCACS=False)
    
    # Create SliceThickness filter for 2.5 mm
    SliceThickness25Filter = DISCHARGEFilter()
    SliceThickness25Filter.createFilter(feature='SliceThickness', exactValue=[2.5], name='SliceThickness25', updateCACS=False)

    # Create site filter
    SiteFilter = DISCHARGEFilter()
    SiteFilter.createFilter(feature='Site', exactValue=['P10', 'P13', 'P29'], updateCACS=False)
    
    # Create Modality Filter
    ModalityFilter = DISCHARGEFilter()
    ModalityFilter.createFilter(feature='Modality', exactValue=['CT'], updateCACS=True)
    
    # Create SliceThickness and PatinetID filter
    SliceThicknessSite0Filter = DISCHARGEFilter()
    SliceThicknessSite0Filter.createFilterJoint(SliceThickness25Filter, SiteFilter, 'AND', name='SliceThickness25Filter AND SiteFilter', updateCACS=False)
    
    SliceThicknessSite1Filter = DISCHARGEFilter()
    SliceThicknessSite1Filter.createFilterJoint(SliceThickness30Filter, SliceThicknessSite0Filter, 'OR', name='(SliceThickness25Filter AND SiteFilter) OR SliceThickness30Filter', updateCACS=True)
    
    # Create SeriesDescription filter
    # SeriesDescriptionFilter = DISCHARGEFilter()
    # def mapFuncS(SeriesDescription):
    #     if type(SeriesDescription) == str:
    #         if 'aidr' in SeriesDescription.lower():
    #             return False
    #         elif 'fbp' in SeriesDescription.lower():
    #             return False
    #         else:
    #             return True
    #     else:
    #         return False
    # SeriesDescriptionFilter.createFilter(feature='SeriesDescription', exactValue=[True], mapFunc=mapFuncS)

    # Create ImageComments CTA word
    ImageCommentsFilter = DISCHARGEFilter()
    def map_ImageComment_CTA(v):
        if type(v) == str:
            return not 'cta' in v.lower()
        else:
            return True
        
    ImageCommentsFilter.createFilter(feature='ImageComments', exactValue=[True], mapFunc=map_ImageComment_CTA, updateCACS=True, name='ImageCommentsCTA')
    
    
    # Append filter
    discharge_filter_list=[]
    discharge_filter_list.append(ModalityFilter)
    discharge_filter_list.append(SliceThickness30Filter)
    discharge_filter_list.append(SliceThickness25Filter)
    discharge_filter_list.append(ReconstructionDiameterFilter)
    discharge_filter_list.append(SliceThicknessSite0Filter)
    discharge_filter_list.append(SliceThicknessSite1Filter)
    #discharge_filter_list.append(SeriesDescriptionFilter)
    discharge_filter_list.append(SiteFilter)
    discharge_filter_list.append(ImageCommentsFilter)
    
    # Create filepath_discharge_filt
    #discharge_filter(filepath_discharge, filepath_discharge_filt, discharge_filter_list)
    
    
'''

Also insgesamt für die Auswahl der non-cardiac structures (NCS) 3 Kriterien:
1. Large FOV (≥32 cm)
2. slice thickness: 1mm (Ausnahme bei GE Scannern: 0.625 mm)
3. Kernel-Info (Lung Kernel + Body Kernel), sodass wir am Ende pro CTA 2 Serien (lung+body) identifizieren. 


 Was den Cut-Off für Large FOV müssten wir "≥32 cm" für alle Scanner definieren
- Definition für slice thickness: 1mm (Ausnahme bei GE Scannern: 0.625 mm) - GE Scanner nutzen folgende Zentren: P10, P13, P29



------------------------------------------
GE Scanner nutzen folgende Zentren: P10, P13, P29
Es müsste die PatientID, StudyInstanceUID, SeriesInstanceUID für die CACS laut ECRF identifiziert werden, bestenfalls für FBP und IR (iter Reko)

FOV: £20 cm (kleiner)
Slice thickness: 3 mm (Ausnahme GE/Philips Scanner: 2.5 mm)
Non-contrast scan (=native CT) CT oder CTA
Verknüpfung mit AIDR/FBP

Questions
- CTA in ProtocolName e.g. row 544

'''
