import os
import cv2
import pandas
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from facenet import Facenet
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def addChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

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
    db_coin_type_list = []
    
    res_dict = {}
    
    print('Start extract db fea---')
    for db in db_list:
        db_coin_type = db.split('_')[0] + '_' + db.split('_')[1]
        if db_coin_type not in db_coin_type_list:
            db_coin_type_list.append(db_coin_type)
        
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

    query_coin_type_list = []
    query_coin_type_predict_list = []
    
    matchedlist = []
    
    for f in file_list:
        similary_dict = dict()
        query_coin_type = f.split('_')[0] + '_' + f.split('_')[1]
        query_coin_type_list.append(query_coin_type)
            
        cur_path = os.path.join(query_dir, f)
        image_query = Image.open(cur_path)
        print('Srart extract query fea---')
        print(f)
        query_fea = model.extract_imagefea(image_query)
        print('End extract query fea---')
        query_img = cv2.imread(cur_path)


        top_padding = 100
        bottom_padding = 200
        left_padding = 100
        right_padding = 100

        padd_height = query_img.shape[0] + top_padding

        res_img = cv2.copyMakeBorder(query_img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        dst_path = res_dir + '/res_' + f
        cv2.imwrite(dst_path, res_img)

        res_img = Image.open(dst_path)
        for dbkey in dbfea_dict:
            dbfea = dbfea_dict[dbkey]
            #distance = np.linalg.norm(query_fea - dbfea, axis=1)
            distance = getCosdis(query_fea[0], dbfea[0])
            similary_dict[dbkey] = float(distance)
        #sortedlist = sorted(similary_dict.items(),key = lambda x:x[1],reverse = False)
        sortedlist = sorted(similary_dict.items(),key = lambda x:x[1],reverse = True)
        #print(sortedlist)
        
        res_dict[f] = sortedlist
        
        index = 0
        match_tag = False
        query_type = f.split('_')

        pre_type = []

        addChineseText(res_img, '真实版别: ' + query_type[0]+'_'+query_type[1], (left_padding,padd_height+10), (0, 0, 0) ,13)
        addChineseText(res_img, '预测版别（按可能性排序）: ', (left_padding,padd_height+30), (0, 0, 0) ,13)
        
        topN_similar_list = []
        for ds in sortedlist:
            if index < topN:
                print(ds)
                db_type = ds[0].split('_')
                if query_type[3] in db_type[3] or db_type[3] in query_type[3]:
                    continue
                pre_cointype = db_type[0]+'_'+db_type[1]
                if pre_cointype not in pre_type:
                    index = index + 1
                    pre_type.append(pre_cointype)
                    topN_similar_list.append(ds[1])
                    print(query_type[3] + '_' + db_type[3])
                    query_coin_type_predict_list.append(db_type[0]+'_'+db_type[1])
                    if (query_type[0]==db_type[0]) and (query_type[1]==db_type[1]):
                        print('Matched: ' + f)
                        match_tag = True
                        addChineseText(res_img, db_type[0]+'_'+db_type[1], (left_padding+10,padd_height+20*(index+3)-10), (255, 0, 0) ,13)
                    else:
                        addChineseText(res_img, db_type[0]+'_'+db_type[1], (left_padding+10,padd_height+20*(index+3)-10), (0, 0, 0) ,13)
                else:
                    continue
        np_topN_similar_array = np.array(topN_similar_list)
        probability = softmax(np_topN_similar_array)
        print(probability)
        if match_tag == True:
            matchedlist.append(f)
            match_num = match_num + 1
        dst_path2 = res_dir + '/ress_' + f
        res_img.save(dst_path2)
        os.remove(dst_path)
    match_ratio = match_num * 1.0 / len(file_list)
    print('match ratio : ' + str(match_ratio))
    
    matchsavefile='query_matched.txt'
    with open(matchsavefile,'w')as file:
        for query in matchedlist:
            file.write(query+'\n')
    
    MAX_OUT = 10
    ressavefile='query_res.txt'
    with open(ressavefile,'w')as file:
        for query in res_dict.keys():
            array_sort = res_dict[query]
            count = 0
            file.write(query+':')
            for ds in array_sort:
                count = count + 1
                if count < MAX_OUT:
                    file.write(ds[0]+'&'+str(ds[1])+'@')
                else:
                    file.write('\n')
                    break
        
    #print(db_coin_type_list)
    #print(query_coin_type_list)
    #print(query_coin_type_predict_list)
    
    #print(len(db_coin_type_list)+':'+len(query_coin_type_list)+':'+len(query_coin_type_predict_list))
    #print("每个类别的精确率和召回率：", classification_report(query_coin_type_list, query_coin_type_predict_list))
 

"""  
    report = classification_report(query_coin_type_list, query_coin_type_predict_list, output_dict=True)
    df =  pandas.DataFrame(report).transpose()
    df.to_excel('report_to_file.xlsx', sheet_name='Sheet1')
    

    sns.set()
    C2= confusion_matrix(query_coin_type_list, query_coin_type_predict_list, labels=db_coin_type_list)
    plt.figure(figsize=(20,16))
    sns.heatmap(C2,annot=True)
    plt.title("Correlations")
    plt.savefig("Correlations.jpg")
"""
