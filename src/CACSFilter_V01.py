# -*- coding: utf-8 -*-
"""
Created: 21.04.2020

@author: Bernhard Foellmer
@version V01

CACSFilter V01 20200421

CACS inclusion criteria
--------
1) ReconstructionDiameter <= 230 mm
2) SliceThickness == 3.0 mm OR (SliceThickness == 2.5mm AND Side == ['P10', 'P13', 'P29'])
3) Modality == 'CT'
4) ImageComments does not include 'cta'

"""

from TagFilter import TagFilter, DISCHARGEFilter

# Set parameter
filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_16042020.xlsx'
#filepath_discharge = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_tags_28022020.xlsx'
filepath_discharge_filt = 'H:/cloud/cloud_data/Projects/CACSFilter/data/discharge_CACS_V01.xlsx'  

# Create ReconstructionDiameter filter
ReconstructionDiameterFilter = DISCHARGEFilter()
ReconstructionDiameterFilter.createFilter(feature='ReconstructionDiameter', mapFunc=lambda v : v <= 230, updateTarget=True, name='ReconstructionDiameter', featFunc=lambda v : v)
# Create SliceThickness filter for 3.0 mm
SliceThickness30Filter = DISCHARGEFilter()
SliceThickness30Filter.createFilter(feature='SliceThickness', mapFunc=lambda v : v == 3.0, updateTarget=False, name='SliceThickness30Filter', featFunc=lambda v : v)
# Create SliceThickness filter for 2.5 mm
SliceThickness25Filter = DISCHARGEFilter()
SliceThickness25Filter.createFilter(feature='SliceThickness', mapFunc=lambda v : v == 2.5, updateTarget=False, name='SliceThickness25Filter', featFunc=lambda v : v)
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
ImageCommentsFilter = DISCHARGEFilter()
ImageCommentsFilter.createFilter(feature='ImageComments', mapFunc=ImageCommentsFilter.includeNotString('cta'), updateTarget=True, name='ImageComments_cta', featFunc=ImageCommentsFilter.includeNotString('cta'))
# Create ProtocolName Calcium Score word
ProtocolNameFilter = DISCHARGEFilter()
ProtocolNameFilter.createFilter(feature='ProtocolName', updateTarget=False, name='ProtocolName_Calcium Score', featFunc=ImageCommentsFilter.includeString('Calcium Score'))
# Create ProtocolName Calcium Score word
SeriesDescriptionFilter = DISCHARGEFilter()
SeriesDescriptionFilter.createFilter(feature='SeriesDescription', updateTarget=False, name='SeriesDescription_CACS', featFunc=ImageCommentsFilter.includeString('CACS'))
# Append filter
discharge_filter_list=[]
discharge_filter_list.append(ModalityFilter)
discharge_filter_list.append(SliceThickness30Filter)
discharge_filter_list.append(SliceThickness25Filter)
discharge_filter_list.append(ReconstructionDiameterFilter)
discharge_filter_list.append(SliceThicknessSite0Filter)
discharge_filter_list.append(SliceThicknessSite1Filter)
discharge_filter_list.append(SiteFilter)
discharge_filter_list.append(ImageCommentsFilter)
discharge_filter_list.append(ProtocolNameFilter)
discharge_filter_list.append(SeriesDescriptionFilter)
# Create filepath_discharge_filt
CACSFilter = TagFilter()
CACSFilter.discharge_filter(filepath_discharge, filepath_discharge_filt, discharge_filter_list, Target='CACS')

    