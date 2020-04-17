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
                df_feature = df[self.feature].apply(self.mapFunc)
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
        
        # Filtr dataframe
        df_filt = filt.filter(df)
        filt.df_filt = df_filt
        
        # Append filter column
        df[filt.name + '_OK'] = df_filt
        
        # Update df_CACS
        if filt.updateCACS:
            df_CACS['CACS'] = df_CACS['CACS'] & df_filt
        
        
    # Compute CACS column
    df_linear = df
    df_linear['CACS'] = df_CACS['CACS']
    
    # Create ordered list
    df_ordered = df_linear.set_index(['PatientID','StudyInstanceUID','SeriesInstanceUID'])
    df_ordered.sort_index(inplace=True)
    
    # Create workbook list    
    writer = pd.ExcelWriter(filepath_discharge_filt)            
    df_to_excel(writer, "ordered", df_ordered)    
    df_to_excel(writer, "linear", df_linear)    
    workbook  = writer.book
    
    # Highlight CACS
    format_red = workbook.add_format({'font_color': 'red'})
    for i in range(df_ordered['CACS'].shape[0]):
        if df_ordered['CACS'][i]:
            first_row=i+1
            last_row=i+1
            first_col=0
            last_col=1000
            print(i, first_row, last_row, first_col, last_col)
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
    filepath_discharge_filt = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags.xlsx'  
    

    # Create ReconstructionDiameter filter
    ReconstructionDiameterFilter = DISCHARGEFilter()
    ReconstructionDiameterFilter.createFilter(feature='ReconstructionDiameter', maxValue=200, updateCACS=True)
    
    # Create SliceThickness filter for 3.0 mm
    SliceThickness30Filter = DISCHARGEFilter()
    SliceThickness30Filter.createFilter(feature='SliceThickness', exactValue=[3.0], name='SliceThickness30', updateCACS=False)
    
    # Create SliceThickness filter for 3.0 mm
    SliceThickness25Filter = DISCHARGEFilter()
    SliceThickness25Filter.createFilter(feature='SliceThickness', exactValue=[2.5], name='SliceThickness25', updateCACS=False)
    
    # # Create PatinetID filter
    # PatinetIDFilter = DISCHARGEFilter()
    # def mapFuncP(PatinetID):
    #     device = PatinetID.split('-')[0]
    #     return device
    # PatinetIDFilter.createFilter(feature='PatientID', exactValue=['10', '13', '29'], mapFunc=mapFuncP)
    
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
    SeriesDescriptionFilter = DISCHARGEFilter()
    def mapFuncS(SeriesDescription):
        if type(SeriesDescription) == str:
            if 'aidr' in SeriesDescription.lower():
                return True
            elif 'fbp' in SeriesDescription.lower():
                return True
            else:
                return False
        else:
            return False
        
    
    
    SeriesDescriptionFilter.createFilter(feature='SeriesDescription', exactValue=[True], mapFunc=mapFuncS)
    
    # Append filter
    discharge_filter_list=[]
    discharge_filter_list.append(ModalityFilter)
    discharge_filter_list.append(SliceThickness30Filter)
    discharge_filter_list.append(SliceThickness25Filter)
    discharge_filter_list.append(ReconstructionDiameterFilter)
    discharge_filter_list.append(SliceThicknessSite0Filter)
    discharge_filter_list.append(SliceThicknessSite1Filter)
    discharge_filter_list.append(SeriesDescriptionFilter)
    discharge_filter_list.append(SiteFilter)
    
    # Create filepath_discharge_filt
    discharge_filter(filepath_discharge, filepath_discharge_filt, discharge_filter_list)
    
    
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



'''
