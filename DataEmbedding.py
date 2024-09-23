import torch
from Bio import SeqIO

def ReadFileFromFasta(filepath):
    seq = []
    for seq_record in SeqIO.parse(filepath, "fasta"):
        seq.append(['>' + seq_record.id.strip(), str(seq_record.seq).strip()])
    return seq

def Tokenization(k=1):
    NA = 'ACGT'
    if k == 1:
        NA_mers = NA
    if k == 2:
        NA_mers = [NA1 + NA2 for NA1 in NA for NA2 in NA]
    if k == 3:
        NA_mers = [NA1 + NA2 + NA3 for NA1 in NA for NA2 in NA for NA3 in NA]
    if k == 4:
        NA_mers = [NA1 + NA2 + NA3 + NA4 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA]
    if k == 5:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA]
    if k == 6:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA]
    if k == 7:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 + NA7 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA for NA7 in NA]
    if k == 8:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 + NA7 + NA8 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA for NA7 in NA for NA8 in NA]

    kmer_dict = dict([i for i in zip(NA_mers, range(1, len(NA_mers) + 1))])

    return kmer_dict

def SeqToToken(fastas, k):

    myDict = Tokenization(k)

    all_sequence = []

    for i in fastas:
        name, sequence = i[0], i[1]
        sequence = sequence.replace('U', 'T')
        # sequence = sequence.replace('T', 'U')
        per_sequence_token = []
        for index in range(0, len(sequence) - k + 1, k):
            vocab = sequence[index:index + k]
            if vocab in myDict.keys():
                token = myDict[vocab]
                per_sequence_token.append(token)

        all_sequence.append(per_sequence_token)

    all_sequence = torch.tensor(all_sequence)

    return all_sequence
