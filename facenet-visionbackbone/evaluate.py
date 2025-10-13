import os
import io
import sys
import csv
import json
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from facenet import Facenet
from torchvision import datasets, transforms

def initExtractorFacenet(modelpath):
    extractor = Facenet(modelpath)
    return extractor

def getCosdis(v1, v2):
    # 如果其中一个是零向量则直接返回
    if np.count_nonzero(v1) == 0 or np.nonzero(v2) == 0:
        return np.nan
    # 求其余弦距离
    cosdis = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    return cosdis

def read_files_in_folder(folder_path, filelist):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            filelist.append(item_path)
        elif os.path.isdir(item_path):
            read_files_in_folder(item_path, filelist)

def extractlistfea(filelist, model, feadict):
    for item_path in tqdm(filelist, desc='extract fea'):
        try:
            coin_id = item_path.split('/')[-2]
            image_db = Image.open(item_path)
            features = extractor.extract_imagefea(image_db)
            feadict[coin_id + '-' + item_path] = features[0]
        except:
            print('fea error')

def get_args_parser():
    parser = argparse.ArgumentParser('evaluation', add_help=False)
    parser.add_argument('--modelpath', default='ep398-loss0.007-val_loss0.042.pth', type=str)
    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    extractor = None
    extractor = initExtractorFacenet(args.modelpath)

    res_dir = input('result xml save dir:')
    db_dir = input('Input coin database dir:')
    query_dir = input('Input query dir:')

    #read and extract db fea
    dbfilelist = []
    dbfea_dict = dict()
    print('Start extract db fea---')
    read_files_in_folder(db_dir, dbfilelist)
    extractlistfea(dbfilelist, extractor, dbfea_dict)
    print('End extract db fea---')

    #read and extract query fea
    queryfilelist = []
    queryfea_dict = dict()
    print('Srart extract query fea---')
    read_files_in_folder(query_dir, queryfilelist)
    extractlistfea(queryfilelist, extractor, queryfea_dict)
    print('End extract query fea---')

    match_num = 0
    pop_num = 0
    pop_match_num = 0
    #topN = int(input('Input topN: '))
    topN = 1

    for query in tqdm(queryfea_dict.keys(), desc='eval'):
        queryfea = queryfea_dict[query]
        similary_dict = dict()
        for dbkey in dbfea_dict.keys():
            if query==dbkey:
                print('self')
            dbfea = dbfea_dict[dbkey]
            #计算cos距离
            distance = getCosdis(queryfea, dbfea)

            # 欧式距离
            #distance = np.linalg.norm(query_fea - dbfea, axis=1)
            similary_dict[dbkey] = float(distance)
        
        # 欧式距离，reverse=False
        #sortedlist = sorted(similary_dict.items(), key=lambda x: x[1], reverse=False)
        # cos距离，reverse=True
        sortedlist = sorted(similary_dict.items(),key = lambda x:x[1],reverse = True)

        query_type = query.split('-')[0].split('_', 2)[-1]

        #top1
        top1 = sortedlist[0]
        top1_type = top1[0].split('-')[0].split('_', 2)[-1]
        #top2
        top2 = sortedlist[1]
        top2_type = top2[0].split('-')[0].split('_', 2)[-1]

        #if float(top1[1]) >= 0.5 or top1_type == top2_type:
        #if float(top1[1]) >= 0.9 or top1_type == top2_type:
        if top1_type == top2_type:
            #top1等于top2,pop+1
            pop_num += 1
            #print('popped: ' + query)
            #if top1_type == query_type:
                #pop_match_num += 1
                #print('popped_matched: ' + query)


        index = -1
        match_tag = False
        for ds in sortedlist:
            index = index + 1
            if index < topN:
                #print(ds)
                db_type = ds[0].split('-')[0].split('_', 2)[-1]
                if query_type == db_type:
                    #print('Matched: ' + query)
                    #query等于db,match上了
                    match_tag = True
            else:
                break
        if match_tag == True:
            match_num += 1

    pop_ratio = pop_num * 1.0 / len(queryfea_dict)
    print('pop ratio : ' + str(pop_ratio))

    #pop_match_ratio = pop_match_num * 1.0 / len(queryfea_dict)
    #print('pop_match ratio : ' + str(pop_match_ratio))

    match_ratio = match_num * 1.0 / len(queryfea_dict)
    print('match ratio : ' + str(match_ratio))
