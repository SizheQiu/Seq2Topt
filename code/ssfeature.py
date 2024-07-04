'''
The code is adopted from https://github.com/Superzchen/iFeature.
'''
import numpy as np
import re

def get_aac(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    output = []
    for aa in AA:
        output.append( seq.count(aa)/len(seq) )     
    return output

def get_aai(seq):
    '''
    Amino acid index
    '''
    return None

def get_ctd(seq):
    return None

def get_CKSAAP(seq):
    return None

def get_dpc(seq):
    '''
    dipeptide composition (DPC)
    '''
    return None

def get_dde(seq):
    '''
    dipeptide deviation from the expected mean (DDE)
    '''
    return None

def get_qso(seq):
    return None

def get_paac(seq):
    '''
    pseudo-amino acid composition (PAAC)
    '''
    return None

def get_ssfeature(seq):
    '''
    Get statistical sequence features: AAC, AAI, CTD,CKSAAP, DPC,DDE, QSO, PAAC.
    '''
    features = []
    return features


