# -*- coding: utf-8 -*-
"""
Created: 27.04.2020

@author: Bernhard Foellmer
@version V02

CACSFilter V02 20200427

CACS inclusion criteria for optimal selection (discharge_filter_opt)
---------------------------------------
1) ReconstructionDiameter <= 230 mm
2) SliceThickness == 3.0 mm OR (SliceThickness == 2.5mm AND Side == ['P10', 'P13', 'P29'])
3) Modality == 'CT'
4) ImageComments does not include 'cta'
4) ImageComments does not include 'cta'

CACS inclusion criteria for alternative selection (discharge_filter_alt01)
---------------------------------------
1) ReconstructionDiameter <= 230 mm
2) SliceThickness == 3.0 mm OR (SliceThickness == 2.5mm AND Side == ['P10', 'P13', 'P29'])
3) Modality == 'CT'
4) ImageComments does not include 'cta'
4) ImageComments does not include 'cta'
"""

from TagFilter import TagFilter, DISCHARGEFilter
from collections import defaultdict

# Set parameter
filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_16042020.xlsx'
#filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_28022020.xlsx'
filepath_discharge_filt = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_CACS_V03.xlsx'  

# Create ReconstructionDiameter filter
ReconstructionDiameterFilter = DISCHARGEFilter()
ReconstructionDiameterFilter.createFilter(feature='ReconstructionDiameter', mapFunc=lambda v : v <= 200, updateTarget=True, name='ReconstructionDiameter', featFunc=lambda v : v)
# Create SliceThickness filter for 3.0 mm
SliceThickness30Filter = DISCHARGEFilter()
SliceThickness30Filter.createFilter(feature='SliceThickness', mapFunc=lambda v : v == 3.0, updateTarget=False, name='SliceThickness30Filter', featFunc=lambda v : v)
# Create SliceThickness filter for 2.5 mm
SliceThickness25Filter = DISCHARGEFilter()
SliceThickness25Filter.createFilter(feature='SliceThickness', mapFunc=lambda v : v == 2.5, updateTarget=False, name='SliceThickness25Filter')
# Create site filter
SiteFilter = DISCHARGEFilter()
SiteFilter.createFilter(feature='Site', mapFunc=lambda v : v in ['P10', 'P13', 'P29'], updateTarget=False, name='SiteFilter', featFunc=lambda v : float(v[1:]))
# Create Modality Filter
ModalityFilter = DISCHARGEFilter()
ModalityFilter.createFilter(feature='Modality', mapFunc=lambda v : v == 'CT', updateTarget=True, name='ModalityFilter', featFunc=lambda v : v == 'CT')
# Create SliceThicknessSite0Filter filter
SliceThicknessSite0Filter = DISCHARGEFilter()
SliceThicknessSite0Filter.createFilterJoin(SliceThickness25Filter, SiteFilter, mapFunc='AND', updateTarget=False, name='SliceThickness25Filter AND SiteFilter')
# Create SliceThicknessSite1Filter filter
SliceThicknessSite1Filter = DISCHARGEFilter()
SliceThicknessSite1Filter.createFilterJoin(SliceThickness30Filter, SliceThicknessSite0Filter, mapFunc='OR', updateTarget=True, name='(SliceThickness25Filter AND SiteFilter) OR SliceThickness30Filter')
# Create ImageComments CTA word
#ImageCommentsFilter = DISCHARGEFilter()
#ImageCommentsFilter.createFilter(feature='ImageComments', mapFunc=DISCHARGEFilter.includeNotString('cta'), updateTarget=True, name='ImageComments_cta', featFunc=ImageCommentsFilter.includeNotString('cta'))
# Create ProtocolName Calcium Score word
ProtocolNameFilter = DISCHARGEFilter()
ProtocolNameFilter.createFilter(feature='ProtocolName', updateTarget=True, name='ProtocolName_Calcium Score', mapFunc=DISCHARGEFilter.includeStringList(
    ['Calcium Score','CaScoring', 'CaScoring','CACS','CaScSeq','SMART SCORE','Calsium score','Calcium Score DISCHARGE','ca score', 'CAVE', 'CALCIUM SCORE']), operation='OR')
# Create ProtocolName Calcium Score word
SeriesDescriptionFilter = DISCHARGEFilter()
#SeriesDescriptionFilter.createFilter(feature='SeriesDescription', updateTarget=False, name='SeriesDescription', featFunc=DISCHARGEFilter.includeString('CACS'))
SeriesDescriptionFilter.createFilter(feature='SeriesDescription', updateTarget=True, name='SeriesDescription', mapFunc=DISCHARGEFilter.includeStringList(
    ['Calcium Score','CaScoring', 'CaScoring','CACS','CaScSeq','SMART SCORE','Calsium score','Calcium Score DISCHARGE','ca score', 'CAVE', 'CALCIUM SCORE']), operation='OR')
# Create ContrastBolusAgent 
ContrastBolusAgentFilter = DISCHARGEFilter()
ContrastBolusAgentFilter.createFilter(feature='ContrastBolusAgent', updateTarget=True, name='ContrastBolusAgent', mapFunc=lambda v : not type(v) == str, featFunc=lambda v : float(not type(v) == str))
# Create CountFilter 
CountFilter = DISCHARGEFilter()
CountFilter.createFilter(feature='Count', updateTarget=True, name='CountFilter', mapFunc=lambda v : (float(v)>30) and (float(v)<90), featFunc=lambda v : v)


# Append filter
discharge_filter_opt=[]
discharge_filter_opt.append(ReconstructionDiameterFilter)
discharge_filter_opt.append(SliceThickness30Filter)
discharge_filter_opt.append(SliceThickness25Filter)
discharge_filter_opt.append(SiteFilter)
discharge_filter_opt.append(SliceThicknessSite0Filter)
discharge_filter_opt.append(SliceThicknessSite1Filter)
discharge_filter_opt.append(ModalityFilter)
discharge_filter_opt.append(ContrastBolusAgentFilter)
discharge_filter_opt.append(CountFilter)
#discharge_filter_opt.append(ProtocolNameFilter)
#discharge_filter_opt.append(SeriesDescriptionFilter)



#####################################################################

# Create ReconstructionDiameter filter
ReconstructionDiameterFilter = DISCHARGEFilter()
ReconstructionDiameterFilter.createFilter(feature='ReconstructionDiameter', mapFunc=lambda v : v <= 300, updateTarget=True, name='ReconstructionDiameter', featFunc=lambda v : v)
# Create SliceThickness filter for 3.0 mm
SliceThickness30Filter = DISCHARGEFilter()
SliceThickness30Filter.createFilter(feature='SliceThickness', mapFunc=lambda v : v == 3.0, updateTarget=False, name='SliceThickness30Filter', featFunc=lambda v : v)
# Create SliceThickness filter for 2.5 mm
SliceThickness25Filter = DISCHARGEFilter()
SliceThickness25Filter.createFilter(feature='SliceThickness', mapFunc=lambda v : v == 2.5, updateTarget=False, name='SliceThickness25Filter')
# Create site filter
SiteFilter = DISCHARGEFilter()
#SiteFilter.createFilter(feature='Site', mapFunc=lambda v : True, updateTarget=False, name='SiteFilter', featFunc=lambda v : float(v[1:]))
SiteFilter.createFilter(feature='Site', mapFunc=lambda v : v in ['P10', 'P13', 'P29'], updateTarget=False, name='SiteFilter', featFunc=lambda v : float(v[1:]))
# Create Modality Filter
ModalityFilter = DISCHARGEFilter()
ModalityFilter.createFilter(feature='Modality', mapFunc=lambda v : v == 'CT', updateTarget=True, name='ModalityFilter', featFunc=lambda v : v == 'CT')
# Create SliceThicknessSite0Filter filter
SliceThicknessSite0Filter = DISCHARGEFilter()
SliceThicknessSite0Filter.createFilterJoin(SliceThickness25Filter, SiteFilter, mapFunc='AND', updateTarget=False, name='SliceThickness25Filter AND SiteFilter')
# Create SliceThicknessSite1Filter filter
SliceThicknessSite1Filter = DISCHARGEFilter()
SliceThicknessSite1Filter.createFilterJoin(SliceThickness30Filter, SliceThicknessSite0Filter, mapFunc='OR', updateTarget=True, name='(SliceThickness25Filter AND SiteFilter) OR SliceThickness30Filter')
# Create ImageComments CTA word
#ImageCommentsFilter = DISCHARGEFilter()
#ImageCommentsFilter.createFilter(feature='ImageComments', mapFunc=DISCHARGEFilter.includeNotString('cta'), updateTarget=True, name='ImageComments_cta', featFunc=ImageCommentsFilter.includeNotString('cta'))
# Create ProtocolName Calcium Score word
ProtocolNameFilter = DISCHARGEFilter()
ProtocolNameFilter.createFilter(feature='ProtocolName', updateTarget=True, name='ProtocolName_Calcium Score', mapFunc=DISCHARGEFilter.includeStringList(
    ['Calcium Score','CaScoring', 'CaScoring','CACS','CaScSeq','SMART SCORE','Calsium score','Calcium Score DISCHARGE','ca score', 'CAVE', 'CALCIUM SCORE']), operation='OR')
# Create ProtocolName Calcium Score word
SeriesDescriptionFilter = DISCHARGEFilter()
#SeriesDescriptionFilter.createFilter(feature='SeriesDescription', updateTarget=False, name='SeriesDescription', featFunc=DISCHARGEFilter.includeString('CACS'))
SeriesDescriptionFilter.createFilter(feature='SeriesDescription', updateTarget=True, name='SeriesDescription', mapFunc=DISCHARGEFilter.includeStringList(
    ['Calcium Score','CaScoring', 'CaScoring','CACS','CaScSeq','SMART SCORE','Calsium score','Calcium Score DISCHARGE','ca score', 'CAVE', 'CALCIUM SCORE']), operation='OR')
# Create ContrastBolusAgent 
ContrastBolusAgentFilter = DISCHARGEFilter()
ContrastBolusAgentFilter.createFilter(feature='ContrastBolusAgent', updateTarget=True, name='ContrastBolusAgent', mapFunc=lambda v : not type(v) == str, featFunc=lambda v : float(not type(v) == str))
# Create CountFilter 
CountFilter = DISCHARGEFilter()
CountFilter.createFilter(feature='Count', updateTarget=True, name='CountFilter', mapFunc=lambda v : (float(v)>=30) and (float(v)<=90), featFunc=lambda v : v)

# Append filter
discharge_filter_alt01=[]
discharge_filter_alt01.append(ReconstructionDiameterFilter)
discharge_filter_alt01.append(SliceThickness30Filter)
discharge_filter_alt01.append(SliceThickness25Filter)
discharge_filter_alt01.append(SiteFilter)
discharge_filter_alt01.append(SliceThicknessSite0Filter)
discharge_filter_alt01.append(SliceThicknessSite1Filter)
discharge_filter_alt01.append(ModalityFilter)
discharge_filter_alt01.append(ContrastBolusAgentFilter)
discharge_filter_alt01.append(CountFilter)
discharge_filter_alt01.append(ProtocolNameFilter)
discharge_filter_alt01.append(SeriesDescriptionFilter)

#####################################################################

# Create filepath_discharge_filt
discharge_targets = []
discharge_targets.append(defaultdict(lambda: None, {'FILTER': discharge_filter_opt, 'TARGET': 'CACS_opt', 'COLOR': 'red'}))
discharge_targets.append(defaultdict(lambda: None, {'FILTER': discharge_filter_alt01, 'TARGET': 'CACS_alt01', 'COLOR': 'blue'}))

# Apply flter
CACSFilter = TagFilter()
self=CACSFilter
CACSFilter.discharge_filter(filepath_discharge, filepath_discharge_filt, discharge_targets)

    