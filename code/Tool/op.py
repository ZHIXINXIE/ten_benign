import pickle,os
import numpy as np
import shutil
import sys
def save(data,filename):
    file = open(filename, 'wb')
    str = pickle.dumps(data)
    file.write(str)
    file.close()

def listdir(path):
    path_list = np.array(os.listdir(path))
    path_list = np.char.add(np.array([path + '/']*len(path_list)),path_list)
    path_list1 = []
    for i in range(len(path_list)):
        path_list1.append(str(path_list[i]))
    return sorted(path_list1)

def load(filename):
    file = open(filename, 'rb')
    a = file.read()
    data = pickle.loads(a)
    return data

def mkdir(path,name = False):
    if not name:
        folder = os.path.exists(path)
        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)
    else:
        folder = os.path.exists(path+'/'+name)
        if not folder:
            os.makedirs(path+'/'+name)
        return path+'/'+name

def extract(li,comp):
    #extract the ele in li which will be fed into function comp with output true
    li1 = []
    for ele in li:
        if comp(ele):
            li1.append(ele)
    return li1

def extract_id(li,comp):
    #extract the index of ele in li which will be fed into function comp with output true
    li1 = []
    for ele in li:
        if comp(ele):
            li1.append(li.index(ele))
    return li1

def name(file):
    return os.path.split(file)[1]

def op_folder(folder,operation):
    #do the same operation to the files in the certain folder
    files = listdir(folder)
    for file in files:
        operation(file)

def copy(old,new,mode = 1):
    if mode:
        shutil.copyfile(old,new)
    else:
        shutil.copytree(old,new)

def minus_list(li1,li2):
    return [ele for ele in li1 if ele not in li2]

def move(old,new):
    shutil.move(old,new)

def op_list(li,comp):
    result = [comp(i) for i in li]
    return result