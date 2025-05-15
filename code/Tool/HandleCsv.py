import pandas as pd

def list2csv(li,filename):
    '''
    Turn a list to a csv file with it name as filename
    :param li: the list
    :param filename: the target csv
    :return: nothing
    '''
    df = pd.DataFrame(li)
    df.to_csv(filename)
def csv2list(filename,format = False):
    '''
    Turn a csv file to a list
    :param filename:csv
    :param format: whether the csv created by pandas(then it has header and index)
    :return: the list returned
    '''
    if format:
        try:
            li = pd.read_csv(filename).values.tolist()
        except :
            return []
        for i in range(0,len(li)):
            del li[i][0]
        return li
    else:
        li = pd.read_csv(filename,header=None).values.tolist()
        return li

def read(filename,m,n):
    '''
    get the value in certain position in certain csv file
    :param filename: the certain csv file
    :param m: the number of the row
    :param n: the number of the colunm
    :return: the value
    '''
    li = csv2list(filename)
    return li[m][n]
def write(filename,m,n,ele):
    '''
    change the value in certain position in certain csv file to certain value
    :param filename: the certain csv file
    :param m: the number of the row
    :param n: the number of the colunm
    :param ele: the certain value
    :return: nothing
    '''
    temp = []
    li = csv2list(filename)
    try:
        li[m][n]=ele
    except IndexError:
        for i in range(len(li)):
            for j in range(n+1-len(li[i])):
                li[i].append([' '])
        for i in range(m+1-len(li)):
            for j in range(n+1):
                temp.append(' ')
            li.append(temp)
            temp = []
        li[m][n]=ele
    list2csv(li,filename)
def clear(filename):
    '''
    clear the csv file with its name as filename
    :param filename: the name of the certain csv
    :return: nothing
    '''
    li = csv2list(filename)
    for i in range(len(li)):
        for j in range(len(li[i])):
            write(filename,i,j,' ')
    list2csv(li,filename)