#coding:utf-8
import pandas as pd
import os

def write_csv(o):
    pic_list = os.listdir('/media/horcham/E/kaggle/Plant Seedlings Classification/test/test')
    name = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']
    index = o.tolist()
    name_list = [name[index[i]] for i in range(len(index))]
    df1 = pd.DataFrame({'file':pic_list,'species':name_list})
    df1 = df1.sort_values(by='file')
    df1.to_csv('submission.csv',index=False)
