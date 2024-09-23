# coding=GBK
import random
import time

import numpy as np
import pandas as pd
from Bio import SeqIO, Seq
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

from DataEmbedding import ReadFileFromFasta, SeqToToken
from Metrics import MetricsCalculate
from ModelConstruction import miProPred_model
from torch import optim
import torch.nn.functional as F
import torch

random_state = 100

k = 2
token_size = 4**k
embedding_dim = 64
num_embeddings = 251 // k
num_heads = 8
n_layers = 4
n_fusionblocks = 4
ffn_num_hiddens = 1024
dropout = 0.2
num_class = 2

lr = 1e-03
epoches = 50
batch_size = 128

cv_fold = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################################
positive_seq = ReadFileFromFasta('data/train_positive_ath_osa_zma_EPD_RGAP_0.8_57865.txt')
negative_seq = ReadFileFromFasta('data/train_negative_ath_osa_zma_cds_random_251nt_0.8_0.8_57865.txt')

positive_independent_seq = ReadFileFromFasta('data/test_positive_SoftBerry_0.8_446.txt')
negative_independent_seq = ReadFileFromFasta('data/test_negative_ath_osa_zma_cds_random_251nt_0.8_0.8_36665.txt')

def train_test_data_split(random_state):

    random.seed(random_state)

    index_list = range(0, len(positive_seq))

    train_sampler = random.sample(index_list, int(len(index_list) * 0.8))
    test_sampler = [i for i in index_list if i not in train_sampler]

    train_positive_seq = []
    train_negative_seq = []
    for i in train_sampler:
        train_positive_seq.append(positive_seq[i])
        train_negative_seq.append(negative_seq[i])

    test_positive_seq = []
    test_negative_seq = []
    for i in test_sampler:
        test_positive_seq.append(positive_seq[i])
        test_negative_seq.append(negative_seq[i])

    train_seq = train_positive_seq + train_negative_seq
    train_label = list(np.ones(len(train_sampler))) + list(np.zeros(len(train_sampler)))

    test1_seq = test_positive_seq + test_negative_seq
    test1_label = list(np.ones(len(test_sampler))) + list(np.zeros(len(test_sampler)))

    test2_seq = positive_independent_seq + negative_independent_seq
    test2_label = list(np.ones(len(positive_independent_seq))) + list(np.zeros(len(negative_independent_seq)))

    return np.array(train_seq), np.array(train_label), np.array(test1_seq), np.array(test1_label), np.array(test2_seq), np.array(test2_label)

def miRNA_promoter_sequence():

    miRNA_promoter_test1 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\\11条大豆\根据TSS位置抽取的promoter序列\gma promoter 11条.txt')
    miRNA_promoter_test2 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\\2022 2005 2020文章中包含的pri-miRNA序列\根据TSS位置抽取的promoter序列\\ath promoter 84条.txt')
    miRNA_promoter_test3 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\mirex2中包含的pri-miRNA序列\根据TSS位置抽取的promoter序列\\ath promoter 297条.txt')
    miRNA_promoter_test4 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\mirex2中包含的pri-miRNA序列\根据TSS位置抽取的promoter序列\\hvu promoter 100条.txt')
    return miRNA_promoter_test1, miRNA_promoter_test2, miRNA_promoter_test3, miRNA_promoter_test4

def miRNA_promoter_error_sequence():
    miRNA_promoter_test1 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\预测错误的序列\\miProPred\\ath promoter 84条-预测正确82条 预测错误2条.txt')
    miRNA_promoter_test2 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\预测错误的序列\\miProPred\\ath promoter 297条-预测正确281条 预测错误16条.txt')
    miRNA_promoter_test3 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\已经知道TSS的miRNA\预测错误的序列\\miProPred\\hvu promoter 100条-预测正确81条 预测错误19条.txt')
    return miRNA_promoter_test1, miRNA_promoter_test2, miRNA_promoter_test3

def train_cv(train_seq, train_label, cv_fold):

    # folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state).split(train_seq, train_label)
    #
    # for i, (train, valid) in enumerate(folds):
    #     if (i + 1) >= 1:
    #
    #         file1 = open("D:\HibinLee\论文写作\第三篇论文\代码\miProPred\data\训练集10折交叉验证划分\\train_seq_fold_" + str(i + 1) + ".txt", "a")
    #         file2 = open("D:\HibinLee\论文写作\第三篇论文\代码\miProPred\data\训练集10折交叉验证划分\\valid_seq_fold_" + str(i + 1) + ".txt", "a")
    #
    #         train_X, train_Y = train_seq[train], train_label[train]
    #         valid_X, valid_Y = train_seq[valid], train_label[valid]
    #
    #         for j in range(len(train_X)):
    #             file1.write(train_X[j][0] + '$' + str(train_Y[j]) + "\n")
    #             file1.write(train_X[j][1] + "\n")
    #
    #         for j in range(len(valid_X)):
    #             file2.write(valid_X[j][0] + '$' + str(valid_Y[j]) + "\n")
    #             file2.write(valid_X[j][1] + "\n")
    #
    #         file1.close()
    #         file2.close()



    file = open("D:\HibinLee\论文写作\第三篇论文\实验结果\\batch_size" + str(batch_size) + "\不同kmer的影响" +"\embedding_size " + str(embedding_dim) + "\\2\\" + str(k) + "mer.txt", "a")

    print("k={} batch_size={} embedding_dim={}".format(k, batch_size, embedding_dim))
    print("running")

    file.write("k=" + str(k) + " batch_size=" + str(batch_size) + " embedding_dim=" + str(embedding_dim) + "\n")
    file.flush()

    folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state).split(train_seq, train_label)

    cv_metrics_value = []

    for i, (train, valid) in enumerate(folds):

        if (i + 1) >= 1:
            print("CV {}".format(i+1))

            file.write("CV " + str(i + 1) + "\n")
            file.flush()

            train_X, train_Y = train_seq[train], train_label[train]
            valid_X, valid_Y = train_seq[valid], train_label[valid]

            train_X = SeqToToken(train_X, k)
            valid_X = SeqToToken(valid_X, k)

            train_ds = TensorDataset(train_X, torch.from_numpy(train_Y).long())
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            valid_ds = TensorDataset(valid_X, torch.from_numpy(valid_Y).long())
            valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)

            model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)

            # model = torch.nn.DataParallel(model)

            opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0005)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
            loss_func = F.cross_entropy

            for epoch in range(1, epoches + 1):

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                y_predict_pro_batch = []
                y_predict_label_batch = []
                valid_Y_batch = []

                train_loss_sum = 0
                val_loss_sum = 0

                model.train()
                for xt, yt in train_dl:
                    train_loss = loss_func(model(xt.to(device)), yt.to(device))
                    train_loss_sum = train_loss_sum + train_loss
                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    for xv, yv in valid_dl:
                        y_predict_pro = model(xv.to(device))
                        val_loss = loss_func(y_predict_pro, yv.to(device))
                        val_loss_sum = val_loss_sum + val_loss

                        y_predict_pro_batch.extend(list(y_predict_pro[:, 1].cpu()))
                        y_predict_label = torch.where(y_predict_pro[:, 1] >= 0.5, 1, 0)
                        y_predict_label_batch.extend(list(y_predict_label.cpu()))
                        valid_Y_batch.extend(list(yv))

                train_loss_avg = float((train_loss_sum / len(train_dl)).cpu())
                val_loss_avg = float((val_loss_sum / len(valid_dl)).cpu())
                print("epoch:{} train----loss:{} val----loss:{}".format(epoch, train_loss_avg, val_loss_avg))
                metrics_value, confusion = MetricsCalculate(valid_Y_batch, y_predict_label_batch, y_predict_pro_batch)
                print("validation data score: ", metrics_value, confusion)

                file.write("epoch:" + str(epoch) + " train----loss:" + str(train_loss_avg) + " val----loss:" + str(val_loss_avg) + "\n")
                file.write("validation data score: " + str(metrics_value) + str(confusion) + "\n")
                file.flush()

                if epoch % epoches == 0:
                    cv_metrics_value.append(metrics_value)

                scheduler.step(val_loss_sum / len(valid_dl))

    print("{}-CV valid_scores: {}".format(cv_fold, np.around(np.array(cv_metrics_value).sum(axis=0) / cv_fold, 3)))
    print("{}-CV valid_scores_std: {}".format(cv_fold, np.around(np.std(np.array(cv_metrics_value), axis=0), 3)))

    file.close()

def train(train_seq, train_label):

    print("k={} batch_size={} embedding_dim={}".format(k, batch_size, embedding_dim))

    print("running")

    train_X = SeqToToken(train_seq, k)
    train_Y = train_label

    train_ds = TensorDataset(train_X, torch.from_numpy(train_Y).long())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    loss_func = F.cross_entropy

    for epoch in range(1, epoches + 1):
        loss_sum = 0
        model.train()
        for xt, yt in train_dl:
            loss = loss_func(model(xt.to(device)), yt.to(device))
            loss_sum = loss_sum + loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch:{} loss:{}".format(epoch, loss_sum / len(train_dl)))
        scheduler.step(loss_sum / len(train_dl))

    torch.save(model.state_dict(), 'model/miProPred_model.pt')

def test_regular(test_seq, test_label):

    test_X = SeqToToken(test_seq, k)
    test_Y = test_label

    test_ds = TensorDataset(test_X, torch.from_numpy(test_Y))
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    y_predict_pro_batch = []
    y_predict_label_batch = []
    test_Y_batch = []

    model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)
    model.load_state_dict(torch.load('model/miProPred_model.pt'))
    model.eval()

    with torch.no_grad():

        for xt, yt in test_dl:
            y_predict_pro = model(xt.to(device))[:, 1]
            y_predict_pro_batch.extend(y_predict_pro.cpu().tolist())

            y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
            y_predict_label_batch.extend(y_predict_label.cpu().tolist())

            test_Y_batch.extend(yt.tolist())

        metrics_value, confusion = MetricsCalculate(test_Y_batch, y_predict_label_batch, y_predict_pro_batch)

        pd.DataFrame(np.column_stack((test_Y_batch, y_predict_pro_batch))).to_csv('./model/testing_dataset2_label_pro.csv', index=False, header=None)

        print("test data score: ", metrics_value, confusion)

# only positive sequence
def test_miRNA_Promoter(test_seq):

    error_index = []

    test_X = SeqToToken(test_seq, k)

    model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)
    model.load_state_dict(torch.load('model/miProPred_model.pt'))
    model.eval()

    with torch.no_grad():
        y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
        y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
        y_predict_label = list(y_predict_label.cpu())

        for index in range(len(y_predict_label)):
            if y_predict_label[index] != 1:
                error_index.append(index)

        true_positive_count = y_predict_label.count(1)

        print("test data score: ", true_positive_count, len(y_predict_label))

        # for elem in error_index:
        #     print(test_seq[elem][0])
        #     print(test_seq[elem][1])

def test_miRNA_Promoter_error(error_seq, species):

    model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)
    model.load_state_dict(torch.load('model/miProPred_model.pt'))
    model.eval()

    count = 0

    if species == 'ath':
        chr_seq = {}
        root_path = 'D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\\ath'
        for seq_record in SeqIO.parse(root_path + "\\TAIR10_chr_all.fas", "fasta"):
            chr_seq[seq_record.id] = seq_record.seq

        for elem in error_seq:
            contents = elem[0].split('||')
            name = contents[0]
            TSS_posi = int(contents[1].split(':')[1])
            chr = contents[2].split(':')[1]
            strand = contents[3].split(':')[1]

            flag = TSS_posi

            if strand == '+':
                while TSS_posi >= flag - 10000:
                    count = count + 1
                    TSS_posi = TSS_posi - 1
                    TSS = chr_seq[chr][TSS_posi - 1]
                    TSS_upstream = chr_seq[chr][TSS_posi - 201:TSS_posi - 1]
                    TSS_downstream = chr_seq[chr][TSS_posi:TSS_posi + 50]
                    core_promoter = str(TSS_upstream + TSS + TSS_downstream)

                    if core_promoter.find('N') == -1:
                        test_X = SeqToToken([['>' + name, core_promoter]], k)
                        with torch.no_grad():
                            y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            y_predict_label = list(y_predict_label.cpu())
                            if y_predict_label.count(1) > 0:
                                break

            if strand == '-':
                while TSS_posi <= flag + 10000:
                    count = count + 1
                    TSS_posi = TSS_posi + 1
                    TSS = chr_seq[chr][TSS_posi - 1]
                    TSS_upstream = chr_seq[chr][TSS_posi - 51:TSS_posi - 1]
                    TSS_downstream = chr_seq[chr][TSS_posi:TSS_posi + 200]
                    core_promoter = TSS_upstream + TSS + TSS_downstream
                    core_promoter = str(Seq.reverse_complement(core_promoter))

                    if core_promoter.find('N') == -1:
                        test_X = SeqToToken([['>' + name, core_promoter]], k)
                        with torch.no_grad():
                            y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            y_predict_label = list(y_predict_label.cpu())
                            if y_predict_label.count(1) > 0:
                                break

    if species == 'hvu':

        for elem in error_seq:
            contents = elem[0].split('||')
            name = contents[0]
            TSS_posi = int(contents[1].split(':')[1])
            chr = contents[2].split(':')[1]
            strand = contents[3].split(':')[1]

            flag = TSS_posi

            chr_seq = {}
            root_path = 'D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\hvu'
            for seq_record in SeqIO.parse(root_path + "\\" + chr + ".fasta", "fasta"):
                chr_seq[seq_record.id] = seq_record.seq

            if strand == '+':
                while TSS_posi >= flag - 10000:
                    count = count + 1
                    TSS_posi = TSS_posi - 1
                    TSS = chr_seq[chr][TSS_posi - 1]
                    TSS_upstream = chr_seq[chr][TSS_posi - 201:TSS_posi - 1]
                    TSS_downstream = chr_seq[chr][TSS_posi:TSS_posi + 50]
                    core_promoter = str(TSS_upstream + TSS + TSS_downstream)

                    if core_promoter.find('N') == -1:
                        test_X = SeqToToken([['>' + name, core_promoter]], k)
                        with torch.no_grad():
                            y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            y_predict_label = list(y_predict_label.cpu())
                            if y_predict_label.count(1) > 0:
                                break

            if strand == '-':
                while TSS_posi <= flag + 10000:
                    count = count + 1
                    TSS_posi = TSS_posi + 1
                    TSS = chr_seq[chr][TSS_posi - 1]
                    TSS_upstream = chr_seq[chr][TSS_posi - 51:TSS_posi - 1]
                    TSS_downstream = chr_seq[chr][TSS_posi:TSS_posi + 200]
                    core_promoter = TSS_upstream + TSS + TSS_downstream
                    core_promoter = str(Seq.reverse_complement(core_promoter))

                    if core_promoter.find('N') == -1:
                        test_X = SeqToToken([['>' + name, core_promoter]], k)
                        with torch.no_grad():
                            y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            y_predict_label = list(y_predict_label.cpu())
                            if y_predict_label.count(1) > 0:
                                break
    print('total sequence: ', len(error_seq))
    print('total distance: ', count)
    print('average distance: ', count / len(error_seq))

def test_regular_Promoter_error1(error_seq):

    model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)
    model.load_state_dict(torch.load('model/miProPred_model.pt'))
    model.eval()

    count1 = 0
    distance1 = 0
    count2 = 0
    distance2 = 0
    count3 = 0
    distance3 = 0

    for elem in error_seq:
        name = str(elem[0])
        sequence = str(elem[1])
        if name.startswith('>AT'):
            for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\\ath\\TAIR10_chr_all.fas", "fasta"):
                index1 = seq_record.seq.find(sequence)
                if index1 != -1:
                    count1 = count1 + 1
                    TSS_posi = index1 + 201

                    flag = TSS_posi

                    while TSS_posi >= flag - 10000:
                        distance1 = distance1 + 1
                        TSS_posi = TSS_posi - 1
                        TSS = seq_record.seq[TSS_posi - 1]
                        TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                        TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                        core_promoter = str(TSS_upstream + TSS + TSS_downstream)

                        if core_promoter.find('N') == -1:
                            test_X = SeqToToken([['>' + name, core_promoter]], k)
                            with torch.no_grad():
                                y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                                y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                                y_predict_label = list(y_predict_label.cpu())
                                if y_predict_label.count(1) > 0:
                                    break
                    break
                if index1 == -1:
                    index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                    if index2 != -1:
                        count1 = count1 + 1
                        TSS_posi = index2 + 51

                        flag = TSS_posi

                        while TSS_posi <= flag + 10000:
                            distance1 = distance1 + 1
                            TSS_posi = TSS_posi + 1
                            TSS = seq_record.seq[TSS_posi - 1]
                            TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                            TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                            core_promoter = TSS_upstream + TSS + TSS_downstream
                            core_promoter = str(Seq.reverse_complement(core_promoter))

                            if core_promoter.find('N') == -1:
                                test_X = SeqToToken([['>' + name, core_promoter]], k)
                                with torch.no_grad():
                                    y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                                    y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                                    y_predict_label = list(y_predict_label.cpu())
                                    if y_predict_label.count(1) > 0:
                                        break
                        break

        if name.startswith('>LOC'):
            flag = 0
            for chr in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\osa\\chr" + chr + ".fasta", "fasta"):
                    index1 = seq_record.seq.find(sequence)
                    if index1 != -1:
                        count2 = count2 + 1
                        TSS_posi = index1 + 201

                        flag = TSS_posi

                        while TSS_posi >= flag - 10000:
                            distance2 = distance2 + 1
                            TSS_posi = TSS_posi - 1
                            TSS = seq_record.seq[TSS_posi - 1]
                            TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                            TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                            core_promoter = str(TSS_upstream + TSS + TSS_downstream)

                            if core_promoter.find('N') == -1:
                                test_X = SeqToToken([['>' + name, core_promoter]], k)
                                with torch.no_grad():
                                    y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                                    y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                                    y_predict_label = list(y_predict_label.cpu())
                                    if y_predict_label.count(1) > 0:
                                        break
                        flag = 1
                        break
                    if index1 == -1:
                        index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                        if index2 != -1:
                            count2 = count2 + 1
                            TSS_posi = index2 + 51

                            flag = TSS_posi

                            while TSS_posi <= flag + 10000:
                                distance2 = distance2 + 1
                                TSS_posi = TSS_posi + 1
                                TSS = seq_record.seq[TSS_posi - 1]
                                TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                                TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                                core_promoter = TSS_upstream + TSS + TSS_downstream
                                core_promoter = str(Seq.reverse_complement(core_promoter))

                                if core_promoter.find('N') == -1:
                                    test_X = SeqToToken([['>' + name, core_promoter]], k)
                                    with torch.no_grad():
                                        y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                                        y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                                        y_predict_label = list(y_predict_label.cpu())
                                        if y_predict_label.count(1) > 0:
                                            break
                            flag = 1
                            break

                if flag == 1:
                    break
        if name.startswith('>GR'):
            flag = 0
            for chr in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\zma\\chromosome" + chr + ".fasta", "fasta"):
                    index1 = seq_record.seq.find(sequence)
                    if index1 != -1:
                        count3 = count3 + 1
                        TSS_posi = index1 + 201

                        flag = TSS_posi

                        while TSS_posi >= flag - 10000:
                            distance3 = distance3 + 1
                            TSS_posi = TSS_posi - 1
                            TSS = seq_record.seq[TSS_posi - 1]
                            TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                            TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                            core_promoter = str(TSS_upstream + TSS + TSS_downstream)

                            if core_promoter.find('N') == -1:
                                test_X = SeqToToken([['>' + name, core_promoter]], k)
                                with torch.no_grad():
                                    y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                                    y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                                    y_predict_label = list(y_predict_label.cpu())
                                    if y_predict_label.count(1) > 0:
                                        break
                        flag = 1
                        break
                    if index1 == -1:
                        index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                        if index2 != -1:
                            count3 = count3 + 1
                            TSS_posi = index2 + 51

                            flag = TSS_posi

                            while TSS_posi <= flag + 10000:
                                distance3 = distance3 + 1
                                TSS_posi = TSS_posi + 1
                                TSS = seq_record.seq[TSS_posi - 1]
                                TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                                TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                                core_promoter = TSS_upstream + TSS + TSS_downstream
                                core_promoter = str(Seq.reverse_complement(core_promoter))

                                if core_promoter.find('N') == -1:
                                    test_X = SeqToToken([['>' + name, core_promoter]], k)
                                    with torch.no_grad():
                                        y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                                        y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                                        y_predict_label = list(y_predict_label.cpu())
                                        if y_predict_label.count(1) > 0:
                                            break
                            flag = 1
                            break

                if flag == 1:
                    break
    count = count1 + count2 + count3
    distance = distance1 + distance2 + distance3
    print('total sequence: ', len(error_seq))
    print('discovered sequence: ', count, count1, count2, count3)
    print('total distance: ', distance, distance1, distance2, distance3)
    print('average distance: ', distance / count)

def test_regular_Promoter_error2(error_seq):

    model = miProPred_model(token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class).to(device)
    model.load_state_dict(torch.load('model/miProPred_model.pt'))
    model.eval()

    count1 = 0
    distance1 = 0

    count2 = 0
    distance2 = 0

    count3 = 0
    distance3 = 0

    count4 = 0
    distance4 = 0

    count5 = 0
    distance5 = 0

    for elem in error_seq:
        name = str(elem[0])
        sequence = str(elem[1])
        if name.find('Arabidopsis') != -1:
            for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\\ath\\TAIR10_chr_all.fas", "fasta"):
                index1 = seq_record.seq.find(sequence)
                if index1 != -1:
                    print(name)
                    # count1 = count1 + 1
                    # TSS_posi = index1 + 201

                    # flag = TSS_posi

                    # while TSS_posi >= flag - 10000:
                    #     distance1 = distance1 + 1
                    #     TSS_posi = TSS_posi - 1
                    #     TSS = seq_record.seq[TSS_posi - 1]
                    #     TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                    #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                    #     core_promoter = str(TSS_upstream + TSS + TSS_downstream)
                    #
                    #     if core_promoter.find('N') == -1:
                    #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                    #         with torch.no_grad():
                    #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                    #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                    #             y_predict_label = list(y_predict_label.cpu())
                    #             if y_predict_label.count(1) > 0:
                    #                 break
                    break
                if index1 == -1:
                    index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                    if index2 != -1:
                        print(name)
                        # count1 = count1 + 1
                        # TSS_posi = index2 + 51

                        # flag = TSS_posi

                        # while TSS_posi <= flag + 10000:
                        #     distance1 = distance1 + 1
                        #     TSS_posi = TSS_posi + 1
                        #     TSS = seq_record.seq[TSS_posi - 1]
                        #     TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                        #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                        #     core_promoter = TSS_upstream + TSS + TSS_downstream
                        #     core_promoter = str(Seq.reverse_complement(core_promoter))
                        #
                        #     if core_promoter.find('N') == -1:
                        #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                        #         with torch.no_grad():
                        #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                        #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                        #             y_predict_label = list(y_predict_label.cpu())
                        #             if y_predict_label.count(1) > 0:
                        #                 break
                        break
        if name.find('Oryza_sativa') != -1:
            flag = 0
            for chr in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\osa\\chr" + chr + ".fasta", "fasta"):
                    index1 = seq_record.seq.find(sequence)
                    if index1 != -1:
                        print(name)
                        # count2 = count2 + 1
                        # TSS_posi = index1 + 201

                        # flag = TSS_posi

                        # while TSS_posi >= flag - 10000:
                        #     distance2 = distance2 + 1
                        #     TSS_posi = TSS_posi - 1
                        #     TSS = seq_record.seq[TSS_posi - 1]
                        #     TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                        #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                        #     core_promoter = str(TSS_upstream + TSS + TSS_downstream)
                        #
                        #     if core_promoter.find('N') == -1:
                        #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                        #         with torch.no_grad():
                        #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                        #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                        #             y_predict_label = list(y_predict_label.cpu())
                        #             if y_predict_label.count(1) > 0:
                        #                 break
                        # flag = 1
                        break
                    if index1 == -1:
                        index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                        if index2 != -1:
                            print(name)
                            # count2 = count2 + 1
                            # TSS_posi = index2 + 51

                            # flag = TSS_posi

                            # while TSS_posi <= flag + 10000:
                            #     distance2 = distance2 + 1
                            #     TSS_posi = TSS_posi + 1
                            #     TSS = seq_record.seq[TSS_posi - 1]
                            #     TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                            #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                            #     core_promoter = TSS_upstream + TSS + TSS_downstream
                            #     core_promoter = str(Seq.reverse_complement(core_promoter))
                            #
                            #     if core_promoter.find('N') == -1:
                            #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                            #         with torch.no_grad():
                            #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            #             y_predict_label = list(y_predict_label.cpu())
                            #             if y_predict_label.count(1) > 0:
                            #                 break
                            # flag = 1
                            break

                if flag == 1:
                    break
        if name.find('Zea_mays') != -1:
            flag = 0
            for chr in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\zma\\chromosome" + chr + ".fasta", "fasta"):
                    index1 = seq_record.seq.find(sequence)
                    if index1 != -1:
                        print(name)
                        # count3 = count3 + 1
                        # TSS_posi = index1 + 201

                        # flag = TSS_posi

                        # while TSS_posi >= flag - 10000:
                        #     distance3 = distance3 + 1
                        #     TSS_posi = TSS_posi - 1
                        #     TSS = seq_record.seq[TSS_posi - 1]
                        #     TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                        #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                        #     core_promoter = str(TSS_upstream + TSS + TSS_downstream)
                        #
                        #     if core_promoter.find('N') == -1:
                        #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                        #         with torch.no_grad():
                        #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                        #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                        #             y_predict_label = list(y_predict_label.cpu())
                        #             if y_predict_label.count(1) > 0:
                        #                 break
                        # flag = 1
                        break
                    if index1 == -1:
                        index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                        if index2 != -1:
                            print(name)
                            # count3 = count3 + 1
                            # TSS_posi = index2 + 51

                            # flag = TSS_posi

                            # while TSS_posi <= flag + 10000:
                            #     distance3 = distance3 + 1
                            #     TSS_posi = TSS_posi + 1
                            #     TSS = seq_record.seq[TSS_posi - 1]
                            #     TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                            #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                            #     core_promoter = TSS_upstream + TSS + TSS_downstream
                            #     core_promoter = str(Seq.reverse_complement(core_promoter))
                            #
                            #     if core_promoter.find('N') == -1:
                            #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                            #         with torch.no_grad():
                            #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            #             y_predict_label = list(y_predict_label.cpu())
                            #             if y_predict_label.count(1) > 0:
                            #                 break
                            # flag = 1
                            break

                if flag == 1:
                    break
        if name.find('Hordeum_vulgare') != -1:
            flag = 0
            for chr in ['1', '2', '3', '4', '5', '6', '7']:
                for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\hvu\\chromosome" + chr + ".fasta", "fasta"):
                    index1 = seq_record.seq.find(sequence)
                    if index1 != -1:
                        print(name)
                        # count4 = count4 + 1
                        # TSS_posi = index1 + 201

                        # flag = TSS_posi

                        # while TSS_posi >= flag - 10000:
                        #     distance4 = distance4 + 1
                        #     TSS_posi = TSS_posi - 1
                        #     TSS = seq_record.seq[TSS_posi - 1]
                        #     TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                        #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                        #     core_promoter = str(TSS_upstream + TSS + TSS_downstream)
                        #
                        #     if core_promoter.find('N') == -1:
                        #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                        #         with torch.no_grad():
                        #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                        #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                        #             y_predict_label = list(y_predict_label.cpu())
                        #             if y_predict_label.count(1) > 0:
                        #                 break
                        # flag = 1
                        break
                    if index1 == -1:
                        index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                        if index2 != -1:
                            print(name)
                            # count4 = count4 + 1
                            # TSS_posi = index2 + 51

                            # flag = TSS_posi

                            # while TSS_posi <= flag + 10000:
                            #     distance4 = distance4 + 1
                            #     TSS_posi = TSS_posi + 1
                            #     TSS = seq_record.seq[TSS_posi - 1]
                            #     TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                            #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                            #     core_promoter = TSS_upstream + TSS + TSS_downstream
                            #     core_promoter = str(Seq.reverse_complement(core_promoter))
                            #
                            #     if core_promoter.find('N') == -1:
                            #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                            #         with torch.no_grad():
                            #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            #             y_predict_label = list(y_predict_label.cpu())
                            #             if y_predict_label.count(1) > 0:
                            #                 break
                            # flag = 1
                            break

                if flag == 1:
                    break
        if name.find('Glycine_max') != -1:
            flag = 0
            for chr in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']:
                for seq_record in SeqIO.parse("D:\HibinLee\论文写作\第三篇论文\数据\miRNA启动子\公共序列\基因组序列\gma\\chromosome" + chr + ".fasta", "fasta"):
                    index1 = seq_record.seq.find(sequence)
                    if index1 != -1:
                        print(name)
                        # count5 = count5 + 1
                        # TSS_posi = index1 + 201

                        # flag = TSS_posi

                        # while TSS_posi >= flag - 10000:
                        #     distance5 = distance5 + 1
                        #     TSS_posi = TSS_posi - 1
                        #     TSS = seq_record.seq[TSS_posi - 1]
                        #     TSS_upstream = seq_record.seq[TSS_posi - 201:TSS_posi - 1]
                        #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 50]
                        #     core_promoter = str(TSS_upstream + TSS + TSS_downstream)
                        #
                        #     if core_promoter.find('N') == -1:
                        #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                        #         with torch.no_grad():
                        #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                        #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                        #             y_predict_label = list(y_predict_label.cpu())
                        #             if y_predict_label.count(1) > 0:
                        #                 break
                        # flag = 1
                        break
                    if index1 == -1:
                        index2 = seq_record.seq.find(Seq.reverse_complement(sequence))
                        if index2 != -1:
                            print(name)
                            # count5 = count5 + 1
                            # TSS_posi = index2 + 51

                            # flag = TSS_posi

                            # while TSS_posi <= flag + 10000:
                            #     distance5 = distance5 + 1
                            #     TSS_posi = TSS_posi + 1
                            #     TSS = seq_record.seq[TSS_posi - 1]
                            #     TSS_upstream = seq_record.seq[TSS_posi - 51:TSS_posi - 1]
                            #     TSS_downstream = seq_record.seq[TSS_posi:TSS_posi + 200]
                            #     core_promoter = TSS_upstream + TSS + TSS_downstream
                            #     core_promoter = str(Seq.reverse_complement(core_promoter))
                            #
                            #     if core_promoter.find('N') == -1:
                            #         test_X = SeqToToken([['>' + name, core_promoter]], k)
                            #         with torch.no_grad():
                            #             y_predict_pro = model(torch.tensor(test_X).to(device))[:, 1]
                            #             y_predict_label = torch.where(y_predict_pro >= 0.5, 1, 0)
                            #             y_predict_label = list(y_predict_label.cpu())
                            #             if y_predict_label.count(1) > 0:
                            #                 break
                            # flag = 1
                            break

                if flag == 1:
                    break

    # count = count1 + count2 + count3 + count4 + count5
    # distance = distance1 + distance2 + distance3 + distance4 + distance5
    # print('total sequence: ', len(error_seq))
    # print('discovered sequence: ', count, count1, count2, count3, count4, count5)
    # print('total distance: ', distance, distance1, distance2, distance3, distance4, distance5)
    # print('average distance: ', distance / count)

if __name__ == '__main__':

    print("starttime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()

    train_seq, train_label, test1_seq, test1_label, test2_seq, test2_label = train_test_data_split(random_state)

    # train_cv(train_seq, train_label, cv_fold)
    # train(train_seq, train_label)

    # test_regular(test1_seq, test1_label)
    test_regular(test2_seq, test2_label)

    # miRNA_promoter_test1, miRNA_promoter_test2, miRNA_promoter_test3, miRNA_promoter_test4 = miRNA_promoter_sequence()
    # test_miRNA_Promoter(miRNA_promoter_test1)
    # test_miRNA_Promoter(miRNA_promoter_test2)
    # test_miRNA_Promoter(miRNA_promoter_test3)
    # test_miRNA_Promoter(miRNA_promoter_test4)

    # miRNA_promoter_test1, miRNA_promoter_test2, miRNA_promoter_test3 = miRNA_promoter_error_sequence()
    # test_miRNA_Promoter_error(miRNA_promoter_test1, 'ath')
    # test_miRNA_Promoter_error(miRNA_promoter_test2, 'ath')
    # test_miRNA_Promoter_error(miRNA_promoter_test3, 'hvu')

    # miRNA_promoter_test1 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\训练集测试集\独立测试集预测错误的序列_只针对正样本\\独立测试集1全部正样本-11573条.txt')
    # test_regular_Promoter_error1(miRNA_promoter_test1)

    # miRNA_promoter_test2 = ReadFileFromFasta('D:\HibinLee\论文写作\第三篇论文\数据\训练集测试集\独立测试集预测错误的序列_只针对正样本\\独立测试集2全部正样本-446条.txt')
    # test_regular_Promoter_error2(miRNA_promoter_test2)

    print(time.time() - start)
    print("endtime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
