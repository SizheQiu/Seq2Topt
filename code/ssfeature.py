'''
The code is adopted from https://github.com/Superzchen/iFeature and https://sourceforge.net/projects/pydpicao/.
'''
import numpy as np
import pandas as pd
import re
import math
from collections import Counter

def get_aac(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    output = {}
    for aa in AA:
        output['AAC_'+aa] = seq.count(aa)/len(seq)
    return output

def get_dpc(seq):
    '''
    dipeptide composition (DPC)
    '''
    output = {}
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    DPs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    for dp in DPs:
        output['DPC_'+ dp] = seq.count(dp)/(len(seq)-1)
    return output

def get_CTriad_nogap(seq):
    '''
    Conjoint Triad Descriptors(CTriad), gap = 0
    '''
    if len(seq) < 3:
        return None
    output = {}
    AAGroup = {'g1': 'AGV','g2': 'ILFP','g3': 'YMTS','g4': 'HNQW',
               'g5': 'RK','g6': 'DE','g7': 'C'}
    myGroups = sorted(AAGroup.keys()); AADict = {};
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g
    features = [f1 + '_'+ f2 + '_' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]
    for f in features:
        output[f] = 0
    for i in range(len(seq)):
        if i+2<len(seq):
            fea = AADict[seq[i]] + '_' + AADict[seq[i+1]]+'_'+AADict[seq[i+2]]
            output[fea] = output[fea] + 1
            
    maxValue, minValue = max(output.values()), min(output.values())
    for f in features:
        output[f] = (output[f] - minValue) / maxValue
    
    return output


# def get_dde(seq):
#     '''
#     dipeptide deviation from the expected mean (DDE)
#     '''
#     myCodons = {'A': 4,'C': 2,'D': 2,'E': 2,'F': 2,'G': 4,'H': 2,'I': 3,'K': 2,'L': 6,'M': 1,
#         'N': 2,'P': 4,'Q': 2,'R': 6,'S': 6,'T': 4,'V': 4,'W': 1,'Y': 2}
#     AA = 'ACDEFGHIKLMNPQRSTVWY'
#     DPs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
#     myTM = []
#     for dp in DPs:
#         myTM.append((myCodons[dp[0]] / 61) * (myCodons[dp[1]] / 61))
#     AADict = {}
#     for i in range(len(AA)):
#         AADict[AA[i]] = i
#     tmpCode = [0] * 400
#     for i in range(len(seq) - 2 + 1):
#         tmpCode[AADict[seq[i]] * 20 + AADict[seq[i+1]]] = tmpCode[AADict[seq[i]] * 20 + AADict[seq[i+1]]] +1
#     if sum(tmpCode) != 0:
#         tmpCode = [x/sum(tmpCode) for x in tmpCode]  
#     myTV = []
#     for i in range(len(myTM)):
#         myTV.append(myTM[i] * (1-myTM[i]) / (len(seq) - 1))
#     for i in range(len(tmpCode)):
#         tmpCode[i] = (tmpCode[i] - myTM[i]) / math.sqrt(myTV[i])
    
#     output = {'DDE_'+DPs[i]:tmpCode[i] for i in range(len(myTM))}
#     return output

# def get_qso(seq):
#     '''
#     Quasi-sequence order.
#     '''
#     output = GetQuasiSequenceOrder(seq)
#     output.update( GetSequenceOrderCouplingNumberTotal(seq) )
#     return output

# def get_ctd(seq):
#     '''
#     Composition,Transition,Distribution
#     '''
#     return CalculateCTD(seq)

# def get_paac(seq):
#     '''
#     Pseudo amino acid composition
#     '''
#     return GetPseudoAAC(seq)

# def get_ssfeature(seq):
#     '''
#     Get statistical sequence features: AAC, CTD,CTriad,DPC,DDE, QSO
#     '''
#     if 'X' in seq:
#         most_aa = Counter(seq).most_common()[0][0]
#         if most_aa == 'X':
#             most_aa = Counter(seq).most_common()[1][0]
#         seq = seq.replace('X',most_aa)
#     results = {}
#     results.update( get_aac(seq) )
#     results.update( get_dpc(seq) )
#     results.update( get_CTriad_nogap(seq) )
#     results.update( get_dde(seq) )
#     results.update( get_qso(seq) )
#     results.update( get_ctd(seq) )
#     return results

# def get_ssf_table(table, target_column, seq_column='sequence'):
#     data = []
#     for i in range(len(table.index)):
#         temp = {target_column: list(table[target_column])[i] }
#         temp.update( get_ssfeature( list(table[seq_column])[i] ) )
#         data.append( temp )
#     result = pd.DataFrame(data)
#     return result
    
    
    
    
    

