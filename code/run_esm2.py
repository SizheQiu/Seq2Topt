import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
from functions import *


def csv2fasta( in_path, out_path ):
    table = pd.read_csv( in_path )
    new_records = [ SeqRecord(Seq( list(table['seq'])[i].strip() ), id = list(table['UniProtID'])[i], \
            name="",description="") for i in table.index ]
    SeqIO.write(new_records, out_path ,"fasta")
    return out_path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='Run ESM-2: --model: name of pretrained model, default=esm2_t33_650M_UR50D;\
                                --input: input table; --output: output directory for extracted representations;\
                                --repr_layers: layers indices from which to extract representations, default=33.' )
    parser.add_argument('--model', default = 'esm2_t33_650M_UR50D', type=str)
    parser.add_argument('--input', type=str, required = True)
    parser.add_argument('--output', type=str , required = True)
    parser.add_argument('--repr_layers', default = 33, type=int )
    out_fasta = str( os.path.dirname( str(args.input) ) )+str( os.path.basename(str(args.input)) ).split('.')[0] +'.fasta'
    csv2fasta( str(args.input), out_fasta  )

    cmd = 'esm-extract ' + str(args.model) + ' ' + out_fasta + ' ' + str(args.output)
    cmd += ' --repr_layers' + str(args.repr_layers) + ' --include'
    run(cmd)
    print(cmd)
    print('Completed!')

