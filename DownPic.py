# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:04:53 2018
cite https://blog.csdn.net/qq_32166627/article/details/60882964
@author: Lee
"""
url='https://image.baidu.com/search/acjson'

import urllib.request,io,os,sys,re,requests
import datetime
import time

def getManyPages(keyword,pages,begin=0):
    params=[]
    for i in range(30+begin,30*pages+30+begin,30):
        params.append({
                'tn': 'resultjson_com',
                'ipn': 'rj',
                'ct': 201326592,
                'is': '',
                'fp':'result',
                'queryWord': keyword,
                'cl': 2,
                'lm': -1,
                'ie': 'utf-8',
                'oe': 'utf-8',
                'adpicid':'', 
                'st': -1,
                'z': '',
                'ic': 0,
                'word': keyword,
                's':'', 
                'se':'', 
                'tab':'', 
                'width':'', 
                'height':'', 
                'face': 0,
                'istype': 2,
                'qc':'', 
                'nc': 1,
                'fr':'',
                'pn': i,
                'rn': 30,
                'gsm': '3c',
                '1528010657715':'', 
                })
    urls=[]
    for i in params:
        try:
            a=requests.get(url,params=i).json()
            #print(a)
            '''
            for i in a.keys():
                if(isinstance(a[i], str)):
                    print('..............')
                    a[i]=re.sub(r"\\x26([a-zA-Z]{0,9});", r"&\1;", a[i]);
                    '''
            print('holy shit..........')
            print(i)
            urls.append(a.get("data"))
        except:
            print('bad...........')
    return urls
def getImg(dataList,localPath):
    if not os.path.exists(localPath):#create new fold
        os.mkdir(localPath)
    x=0
    
    for i in dataList:
        for j in i:
            if j.get('thumbURL')!=None:
                print('downing... %s'%j.get('thumbURL'))
                ir=requests.get(j.get('thumbURL'))
                nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                open(localPath+nowTime+'%d.jpg'%x,'wb').write(ir.content)
                x+=1
            else:
                print('Image does not exist')
if __name__=='__main__':
    dataList=getManyPages('手型',30,0)#x*30
    getImg(dataList,'D:\\xuexiziliao\\Proj\\CNN_V1.0\\V1_0\\IMG\Phone\\')   