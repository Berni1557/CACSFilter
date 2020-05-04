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
from sklearn.metrics import confusion_matrix
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
        self.filter0 = ''
        self.filter1 = ''
        self.mapFunc = None
        self.featFunc = None
        self.color = ''
        self.df_filt=None
        self.updateTarget=True
        self.operation = 'AND' # 'AND' / 'OR'
        
    def createFilter(self, feature='', minValue=None, maxValue=None, exactValue=[], mapFunc=None, name='', color='', updateTarget=True, featFunc=None, operation ='AND'):
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
        self.mapFunc = mapFunc
        self.featFunc = featFunc
        self.color = color
        self.updateTarget = updateTarget
        if name:
            self.name = name
        else:
            self.name = feature
        self.operation = operation
                
    def createFilterJoin(self, filter0, filter1, mapFunc, name='FilterJoint', updateTarget=True, featFunc=None, operation ='AND'):
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
        
        self.filtetype = 'JOIN'
        self.filter0 = filter0
        self.filter1 = filter1
        self.name = name
        self.updateTarget = updateTarget
        self.mapFunc = mapFunc
        self.featFunc = featFunc
        self.operation = operation
    
    @staticmethod
    def includeString(s):
        def func(v):
            if type(v) == str:
                return s in v.lower()
            else:
                return False
        return func
    
    @staticmethod
    def includeNotString(s):
        def func(v):
            if type(v) == str:
                return not (s in v.lower())
            else:
                return True
        return func
        
    @staticmethod
    def includeStringList(sList):
        def func(v):
            if type(v) == str:
                for s in sList:
                    if s in v.lower():
                        return True
                return False
            else:
                return False
        return func

    @staticmethod
    def includeNotStringList(sList):
        def func(v):
            if type(v) == str:
                for s in sList:
                    if (s in v.lower()):
                        return False
                return True
            else:
                return True
        return func
    
    def filter(self, df):
        """ Filter dataframe df
        
        :param df: Pandas dataframe
        :type df: pd.Dataframe
        """

        if self.filtetype=='FILTER':
            df_filt = df[self.feature].apply(self.mapFunc)
            return df_filt
        elif self.filtetype=='JOIN':
            if self.mapFunc=='AND':
                df_filt0 = self.filter0.filter(df)
                df_filt1 = self.filter1.filter(df)
                df_filt = df_filt0 & df_filt1
                return df_filt
            elif self.mapFunc=='OR':
                df_filt0 = self.filter0.filter(df)
                df_filt1 = self.filter1.filter(df)
                df_filt = df_filt0 | df_filt1
                return df_filt
            else:
                raise ValueError('Filtetype: ' + self.filtetype + ' does not exist.')
        else:
            raise ValueError('Filtetype: ' + self.filtetype + ' does not exist.')
        return None
    
    def __str__(self):
        """ print filer (e.g. print(filter))
        """
        out = ['Filter:']
        out.append('Feature: ' + self.feature)
        out.append('Name: ' + self.name)
        return '\n'.join(out)

class TagFilter:
    """ Create TagFilter
    Class to filter tags
    """

    def confidencePredictor(self, df_linear, discharge_filter_list, Target = 'CACS'):
        """ Calculate fonfidence score using random forest classifier
        
        :param df_linear: Dataframe of the data
        :type df_linear: pd.Dataframe
        :param discharge_filter_list: List of DISCHARGEFilter
        :type discharge_filter_list: list
        :param Target: Column name of the target
        :type Target: str
        """
        
        df = df_linear.copy()
        sheet = 'linear'
        
        # Define features
        df_features = pd.DataFrame()
        for filt in discharge_filter_list:
            if filt.featFunc:
                df_f = df[filt.feature].apply(filt.featFunc)
                df_f = df_f.rename(filt.feature)
                df_features = pd.concat([df_features, df_f], axis=1)
        
        # Replace True and False with zero and one
        df_features = df_features*1
    
        # Read label
        X = np.array(df_features)
        Y = np.array(df[Target])
        
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
        df_confidence = pd.DataFrame(confidence, columns=[Target + '_confidence'])
        
        return df_confidence, C, ACC
          
        
    def discharge_filter(self, filepath_discharge, filepath_discharge_filt, discharge_targets=[]):
        """ Filter DISCHARGE excel sheet
        
        :param filepath_discharge: Filpath of the input DISCHARGE excel sheet
        :type filepath_discharge: str
        :param filepath_discharge_filt: Filpath of the output DISCHARGE excel sheet with boolean columns from filters
        :type filepath_discharge_filt: str
        :param discharge_filter_list: List of DISCHARGEFilter
        :type discharge_filter_list: list
        :param Target: Column name of the target
        :type Target: str
        """
        
        # defaultdict(lambda: None, {'FILTER': discharge_filter_alt01, 'TARGET': 'CACS_alt01', 'COLOR': 'blue'})
        
        # Read discharge tags from linear sheet
        print('Reading file', filepath_discharge)
        sheet = 'linear'
        df = pd.read_excel(filepath_discharge, sheet) 
        
        df_linear = df.copy()
        
        #n = df_linear.shape[0]
        #df_linear = df_linear[n-15:]

        
        # Iterate over discharge_targets and add features
        df_TargetList=[]
        for target in discharge_targets:
            # Iterate over filters
            discharge_filter_list = target['FILTER']
            df_Target = pd.DataFrame(data=True, index=df_linear.index, columns = [target['TARGET']])
            for filt in discharge_filter_list:
                print('Target:', target['TARGET'],'Apply filter:', filt.name, 'Operation', filt.operation)
                # Filtr dataframe
                if filt.mapFunc:
                    df_filt = filt.filter(df_linear)       
                # Append filter column
                df_linear[target['TARGET'] + '_' + filt.name] = df_filt
                
                #print('df_filt', df_filt)
                
                # Update df_target
                if filt.updateTarget:
                    if filt.operation == 'AND':
                        df_Target[target['TARGET']] = df_Target[target['TARGET']] & df_filt
                    elif filt.operation == 'OR':
                        df_Target[target['TARGET']] = df_Target[target['TARGET']] | df_filt
                
                #print('df_Target', df_Target[target['TARGET']])
                
            df_TargetList.append(df_Target)
                
        # # Iterate over discharge_targets and add target column
        for i, target in enumerate(discharge_targets):
            df_linear[target['TARGET']] = df_TargetList[i]
            
            #print('df_Targeti', df_TargetList[i])
            
            # Compute confidence
            df_confidence, C, ACC = self.confidencePredictor(df_linear, discharge_filter_list, Target = target['TARGET'])
            print('Accuracy:', ACC)
            print('Confusion matrix:', C)
            df_linear = pd.concat([df_linear,df_confidence], axis=1)
            
        # Create ordered list
        df_ordered = df_linear.copy()
        df_ordered = df_linear.set_index(['PatientID','StudyInstanceUID','SeriesInstanceUID'])
        df_ordered.sort_index(inplace=True)
        
        # Create workbook list    
        print('Create workbook')
        writer = pd.ExcelWriter(filepath_discharge_filt)            
        df_to_excel(writer, "ordered", df_ordered)    
        df_to_excel(writer, "linear", df_linear)    
        workbook  = writer.book
        
        # Highlight Target
        for target in discharge_targets:
            print('Highlight Target:' + target['TARGET'])
            formatColor = workbook.add_format({'font_color': target['COLOR']})
            print('color', target['COLOR'])
            for i in range(df_ordered[target['TARGET']].shape[0]):
                if df_ordered[target['TARGET']][i]:
                    first_row=i+1
                    last_row=i+1
                    first_col=2
                    last_col=1000
                    writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'no_blanks','format': formatColor})
                    writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'blanks','format': formatColor})
        
            # # Highlight features
            # for filt in discharge_filter_list:
            #     if filt.color:
            #         format_red = workbook.add_format({'font_color': filt.color})
            #         for i in range(filt.df_filt.shape[0]):
            #             if filt.df_filt[i]:
            #                 first_row=i+1
            #                 last_row=i+1
            #                 first_col=0
            #                 last_col=1000
            #                 writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'no_blanks','format': format_red})
            #                 writer.sheets['ordered'].conditional_format(first_row, first_col, last_row, last_col,{'type': 'blanks','format': format_red})
        
        # Add sheet for number of highligted CACS per patient
        columns = ['PatientID']
        for target in discharge_targets:
            columns.append(target['TARGET'] + '_num')
        columns.append('Modality_CT_FOUND')
        columns.append('Confidence_alt01_min')
        
        df_Patient = pd.DataFrame(columns = columns)
        patientList = list(df_linear['PatientID'].unique())
        for p, patient in enumerate(patientList):
            df_pat = df_linear[df_linear['PatientID']==patient]
            NumTarget=[]
            for i, target in enumerate(discharge_targets):
                NumTarget.append((df_pat[target['TARGET']]==True).sum())
            Modality_CT = 'CT' in list(df_pat['Modality'])
            Confidence_alt01 = min(list(df_pat['CACS_alt01_confidence']))
            df_Patient.loc[p] = [patient] + NumTarget + [Modality_CT, Confidence_alt01]
        #df_Patient['Modality_CT'] = df_linear['Modality']
        df_to_excel(writer, "patients", df_Patient)  
            
        # Write excel sheet
        writer.save()
        
        return 
