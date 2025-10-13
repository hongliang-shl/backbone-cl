import os
import cv2
import numpy as np
from PIL import Image
from facenet import Facenet

def getCosdis(v1, v2):
    # 如果其中一个是零向量则直接返回
    if np.count_nonzero(v1) == 0 or np.nonzero(v2) == 0:
        return np.nan
    # 求其余弦距离
    cosdis = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    return cosdis

if __name__ == "__main__":
    model = Facenet()

    db_dir = input('Input coin database dir:')
    db_list = os.listdir(db_dir)
    dbfea_dict = dict()
    print('Start extract db fea---')
    for db in db_list:
        db_path = os.path.join(db_dir, db)
        image_db = Image.open(db_path)
        fea = model.extract_imagefea(image_db)
        dbfea_dict[db] = fea

    print('End extract db fea---')
    query_dir = input('Input query dir:')
    file_list = os.listdir(query_dir)
    res_dir = input('result save dir:')
    match_num = 0
    topN = int(input('Input topN: '))
    
    for f in file_list:
        similary_dict = dict()
        cur_path = os.path.join(query_dir, f)
        image_query = Image.open(cur_path)
        print('Srart extract query fea---')
        print(f)
        query_fea = model.extract_imagefea(image_query)
        print('End extract query fea---')
        for dbkey in dbfea_dict:
            dbfea = dbfea_dict[dbkey]
            #cos距离
            #distance = getCosdis(query_fea[0], dbfea[0])
            #欧式距离
            distance = np.linalg.norm(query_fea - dbfea, axis=1)
            similary_dict[dbkey] = float(distance)
        #欧式距离，reverse=False
        sortedlist = sorted(similary_dict.items(),key = lambda x:x[1],reverse = False)
        
        #cos距离，reverse=True
        #sortedlist = sorted(similary_dict.items(),key = lambda x:x[1],reverse = True)
        index = -1
        match_tag = False
        for ds in sortedlist:
            index = index + 1
            if index < topN:
                print(ds)
                query_type = f.split('_')
                db_type = ds[0].split('_')
                if query_type[0]==db_type[0]:
                    print('Matched: ' + f)
                    match_tag = True
        if match_tag == True:
            match_num = match_num + 1
    match_ratio = match_num * 1.0 / len(file_list)
    print('match ratio : ' + str(match_ratio))
