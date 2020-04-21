#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:40:52 2020

@author: lukass

pip3.8 install xlrd==1.1.0

"""

import numpy as np
import pandas as pd
#import os,shutil
#import xlrd
#from datetime import datetime, date
#from discharge_xa_projections import df_to_excel
#from ecrf_compare import  df_to_excel

def autosize_excel_columns(worksheet, df):
  autosize_excel_columns_df(worksheet, df.index.to_frame())
  autosize_excel_columns_df(worksheet, df, offset=df.index.nlevels)

def autosize_excel_columns_df(worksheet, df, offset=0):
  for idx, col in enumerate(df):
    series = df[col]
    
    #import sys
    #reload(sys)  # Reload does the trick!
    #sys.setdefaultencoding('UTF8')
    
    max_len = max((  series.astype(str).map(len).max(),      len(str(series.name)) )) + 1        
    max_len = min(max_len, 100)
    worksheet.set_column(idx+offset, idx+offset, max_len)
    
def df_to_excel(writer, sheetname, df):
     
    df.to_excel(writer, sheet_name = sheetname, freeze_panes = (df.columns.nlevels, df.index.nlevels))
    autosize_excel_columns(writer.sheets[sheetname], df)
    
def format_differences(worksheet, levels, df_ref, df_bool, format_highlighted):
    
  

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

def discharge_large_fov():
    
    #f = '/home/lukass/Downloads/discharge_tags_16042020.xlsx'
    f = 'H:/cloud/cloud_data/Projects/DL/Code/src/solutions/CACSFilter/discharge_tags_16042020.xlsx'
    df0 = pd.read_excel(f, 'linear') 
    df0.drop('Unnamed: 0',axis=1, inplace=True)
    #'StudyTime',
    df0.drop(columns=['InstanceNumber', 'AcquisitionTime', 'NumberOfFrames'], inplace=True)
    df = df0[df0['Modality']=='CT']    
    #df = df.head(900)
    
    for i in df.index:
        
        size = 0.0 if np.isnan(df.loc[i,'ReconstructionDiameter']) else float(df.loc[i,'ReconstructionDiameter'])
        thick = 0.0 if np.isnan(df.loc[i,'SliceThickness']) else float(df.loc[i,'SliceThickness'])
        id_split = str(df.loc[i,'PatientID']).split('-')        
        site = 'P'+ id_split[0]
        
        sizeok = (size >= 320.0) 
        
        hasge = (site=='P10' or site=='P13' or site=='P29')
        if hasge: 
            thickok = (thick == 0.625)
        else:
            thickok = (thick == 1.0)
            
        islarge = (sizeok and thickok)
        print (i, site, size, thick, hasge)
                 
        df.loc[i,'Site'] = site
        df.loc[i,'HasGE'] = hasge
        df.loc[i,'SizeOK'] = sizeok    
        df.loc[i,'ThicknessOK'] = thickok
        df.loc[i,'IsLarge'] = islarge
        
    ##ff = '/home/lukass/Downloads/large_fov_16042020.xlsx'        
    ff = 'H:/cloud/cloud_data/Projects/DL/Code/src/solutions/CACSFilter/discharge_tags.xlsx'    
    df2 = df.set_index(['PatientID','StudyInstanceUID','SeriesInstanceUID'])
    df2.sort_index(inplace=True)          
    
    writer = pd.ExcelWriter(ff)            
    df_to_excel(writer, "ordered", df2)    
    df_to_excel(writer, "linear", df)    
    workbook  = writer.book
     
    
    df_bool = pd.DataFrame(index=df2.index,columns=df2.columns)
    for i in df2.index:
        if df2.loc[i,'IsLarge']:
            df_bool.loc[i,:] = True 
        else:
            df_bool.loc[i,:] = False
    
    format_red = workbook.add_format()
    #format_red.set_bold()
    format_red.set_font_color('red')
    
    format_differences(writer.sheets['ordered'], df2.index.nlevels, df2, df_bool, format_red)
    
    
    
    
    writer.save()
    
    
    
    return 
    
    
if __name__=='__main__':   

    discharge_large_fov()
    
    
'''

Also insgesamt für die Auswahl der non-cardiac structures (NCS) 3 Kriterien:
1. Large FOV (≥32 cm)
2. slice thickness: 1mm (Ausnahme bei GE Scannern: 0.625 mm)
3. Kernel-Info (Lung Kernel + Body Kernel), sodass wir am Ende pro CTA 2 Serien (lung+body) identifizieren. 


 Was den Cut-Off für Large FOV müssten wir "≥32 cm" für alle Scanner definieren
- Definition für slice thickness: 1mm (Ausnahme bei GE Scannern: 0.625 mm) - GE Scanner nutzen folgende Zentren: P10, P13, P29


'''