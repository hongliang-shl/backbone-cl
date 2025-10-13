import os
import cv2
import numpy as np
from PIL import Image
from facenet import Facenet

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
        query_img = cv2.imread(cur_path)
        res_img = cv2.copyMakeBorder(query_img, 100, 100, 100, 100, cv.BORDER_CONSTANT, value=[0, 255, 0])
        for dbkey in dbfea_dict:
            dbfea = dbfea_dict[dbkey]
            distance = np.linalg.norm(query_fea - dbfea, axis=1)
            similary_dict[dbkey] = float(distance)
        sortedlist = sorted(similary_dict.items(),key = lambda x:x[1],reverse = False)
        #print(sortedlist)
        index = -1
        match_tag = False
        for ds in sortedlist:
            index = index + 1
            if index < topN:
                print(ds)
                query_type = f.split('_')
                db_type = ds[0].split('_')
                if (query_type[0]==db_type[0]) and (query_type[1]==db_type[1]):
                    print('Matched: ' + f)
                    match_tag = True
        if match_tag == True:
            match_num = match_num + 1
    match_ratio = match_num * 1.0 / len(file_list)
    print('match ratio : ' + str(match_ratio))
