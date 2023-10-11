from operator import itemgetter
import torch
from pyfaidx import Fasta
import pandas as pd


def seq2num(seq):
    seq = seq.lower()
    seqdic = {}
    seqdic['a'] = torch.Tensor([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    seqdic['c'] = torch.Tensor([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    seqdic['g'] = torch.Tensor([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    seqdic['t'] = torch.Tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
    seqdic['-'] = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    seqdic['r'] = torch.Tensor([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    seqdic['y'] = torch.Tensor([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    seqdic['m'] = torch.Tensor([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
    seqdic['k'] = torch.Tensor([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
    seqdic['s'] = torch.Tensor([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
    seqdic['w'] = torch.Tensor([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
    seqdic['h'] = torch.Tensor([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    seqdic['b'] = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    seqdic['v'] = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
    seqdic['d'] = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    seqdic['n'] = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    result = torch.stack(itemgetter(*list(seq))(seqdic)).unsqueeze(dim=0)
    return result


genes = Fasta('example.fasta')
genes.keys()

ls_label = []
ls_data = []

df = pd.read_csv('SpeciesName2Label.csv')

for i,row in enumerate(df.iterrows()):
    seqid = row[1]['SampleID']
    label = row[1]['Label']
    ls_data.append(seq2num(genes[seqid][:].seq))
    ls_label.append(torch.tensor(label))

prefix = 'example'
torch.save(ls_label, f'{prefix}_label.pt')
torch.save(ls_data, f'{prefix}_data.pt')