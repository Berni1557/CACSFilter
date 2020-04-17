# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:00:00 2020

@author: Bernhard Foellmer

"""
from collections import defaultdict

class CTSeries:
    """ Create CTSeries
    """
    
    def __init__(self):
        """ Init DISCHARGEFilter
        """
        self.SeriesInstanceUID = None
        self.Count = None
        self.Site = None
        self.Modality = None
        
class CTPatient:
    """ Create CTPatient
    """
    
    def __init__(self):
        """ Init DISCHARGEFilter
        """
        self.PatientID = ''
        self.StudyInstanceUID = ''
        self.Series = []

    return 

if __name__=='__main__':
    
    # Set parameter
    filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_16042020.xlsx'
    filepath_discharge_filt = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags.xlsx'  
    
    patient = CTPatient()

