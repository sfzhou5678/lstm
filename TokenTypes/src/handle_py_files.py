# encoding:utf-8


import os
import hashlib

from src import ast_to_type_with_name

def is_py_file(path):
    if (path.split('.')[-1])=='py':
        return True
    return False

def getFileMD5(filepath):
    try:
        f = open(filepath,'rb')
        md5obj = hashlib.md5()
        md5obj.update(f.read())
        hash = md5obj.hexdigest()
        f.close()
        return str(hash).upper()
    except:
        return None

files_md5_set=set()
def is_duplicated(path):
    global files_md5_set

    md5=getFileMD5(path)
    if md5==None:
        return True
    if md5 in files_md5_set:
        return True
    else:
        files_md5_set.add(md5)
        return False


def handle_py_files(rootdir,wf):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            handle_py_files(path,wf)
        if os.path.isfile(path):
            if is_py_file(path):
                if is_duplicated(path):
                    print('*'*20+path)
                    return
                # print(path)
                # res=base_ast_to_type.file_to_type(path)
                res=ast_to_type_with_name.file_to_type(path)
                if res!='' and res!='Module':
                    wf.write(res+'\n')

rootdir = r'D:\py_project\Tensorflow\myEx'

result_txt_path=r'res.txt'
with open(result_txt_path,'w') as wf:
    handle_py_files(rootdir,wf)
    wf.close()

import random
def SplitData(fname):
    f=open(fname)
    trainfile=open('data/train.txt','w')
    testfile=open('data/test.txt','w')
    code=f.read().split('\n')
    random.shuffle(code)
    splitIndex=int(0.7*len(code))
    trainData=code[:splitIndex]
    testData=code[splitIndex:]

    for i in range(len(trainData)):
        trainfile.write(trainData[i])
        trainfile.write('\n')
    for i in range(len(testData)):
        testfile.write(testData[i])
        testfile.write('\n')
    f.close()
    trainfile.close()
    testfile.close()

SplitData('data/res.txt')