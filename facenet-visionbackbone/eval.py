import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet import Facenet

def prepare_model(chkpt_dir, arch='RETFound_mae'):
    model = Facenet()
    return model

def run_one_image(img, model, arch):
    fea = model.extract_imagefea(img).squeeze()
    #print(fea.shape)
    return fea

def recursive_list_files(directory, filelist):
    for root, dirs, files in os.walk(directory):
        for file in files:
            #print(os.path.join(root, file))
            filelist.append(os.path.join(root, file))

def get_feature(data_path,
                chkpt_dir,
                device,
                arch='RETFound_mae'):
    name_list = []
    feature_list = []
    img_list = []
    recursive_list_files(data_path, img_list)

    #loading model
    model_ = prepare_model(chkpt_dir, arch)

    finished_num = 0
    for i in tqdm(img_list):
        finished_num+=1
        if (finished_num%1000 == 0):
            print(str(finished_num)+"finished")
        #read image
        img = Image.open(i)
        latent_feature = run_one_image(img, model_,arch)
        #combine label and fea
        labels = i.split('/')[-1].split('_')
        label = labels[0] + '_' + labels[1] + '_' + labels[2]
        name_list.append(label)
        feature_list.append(latent_feature)  
    return [name_list,feature_list]

if __name__ == '__main__':
    #model dir
    chkpt_dir = ''
    #data dir
    data_path_gallery = '/mnt/ossfs2-bucket/train-data/kaiyuantongbao_20251013/gallery'
    data_path_query = '/mnt/ossfs2-bucket/train-data/kaiyuantongbao_20251013/query-easy'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    arch='dinov3'   #mambavision/dinov3/RETFound_dinov2

    #extract fea of query and gallery
    [name_list_gallery,feature_gallery_list]=get_feature(data_path_gallery, chkpt_dir, device, arch=arch)
    [name_list_query,feature_query_list]=get_feature(data_path_query, chkpt_dir, device, arch=arch)
    feature_gallery = torch.tensor(feature_gallery_list).to('cuda')
    feature_query = torch.tensor(feature_query_list).to('cuda')

    #计算相似度
    matrix = torch.matmul(feature_query, feature_gallery.T)   #todo: cos dist   
    normA = torch.norm(feature_query, p=2, dim=1).unsqueeze(0)  # 分母
    normB = torch.norm(feature_gallery, p=2, dim=1).unsqueeze(0)  # 分母
    cos = matrix.div(torch.mm(normA.T, normB))
    max_values, max_indices = torch.max(cos, dim=1)

    #save benchmark
    correct = 0
    for i in range(len(name_list_query)):
        if name_list_query[i] == name_list_gallery[max_indices[i]]:
            correct += 1
    with open('benchmark.txt', 'a') as file:
        file.write(chkpt_dir + ',' + '准确率:' + str(correct/len(name_list_query)) + '\n')