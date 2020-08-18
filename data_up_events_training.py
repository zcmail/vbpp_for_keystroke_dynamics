import pandas as pd
import numpy as np

#time：2020年7月30日
#author：ZhangChang

#function description：函数描述
#处理CMU数据集，生产一次事件的数据
#user 是用户名,line_start
#fileName是数据文件名

#测试用例
'''
fileName = "./data/DSL-StrongPasswordData.xls"
keystroke_data,end_time = make_estimate_data('s002',0,fileName)
print (keystroke_data,end_time)
'''
def make_estimate_data_for_up(user,line_start,line_end,fileName):
    df_oral = pd.read_excel(fileName,header=0)
    df = df_oral[df_oral["subject"]==user]
    #df = pd.read_excel(fileName, header=0)
    data_instance = []  # 一次击键行为对应的数据。
    data_instance_all = []
    data_temp_hold = []

    for i in range(line_start, line_end):
        for j in range(11):
            temp_Hole = df.iloc[i, 3 + j * 3]   # Hold的时间。
            data_temp_hold.append(temp_Hole)  # 记录Hold time 时间。
        data_temp_hold.sort()
        data_instance_all.append(data_temp_hold)
        data_temp_hold = []

    return np.concatenate(data_instance_all)

def make_estimate_data_for_up_all(fileName,user):
    df = pd.read_excel(fileName,header=0)
    #df = df_oral[df_oral["subject"]==user]
    #df = pd.read_excel(fileName, header=0)
    data_instance = []  # 一次击键行为对应的数据。
    data_instance_all = []
    data_temp_hold = []

    for i in range(0, 20400):
        #跳过user
        if df.iloc[i,0] == user:
            continue
        # 只取每个用户的前50个样本
        if i%400 >= 50:
            continue
        for j in range(11):
            temp_Hole = df.iloc[i, 3 + j * 3]   # Hold的时间。
            data_temp_hold.append(temp_Hole)  # 记录Hold time 时间。
        data_temp_hold.sort()
        data_instance_all.append(data_temp_hold)
        data_temp_hold = []

    return np.concatenate(data_instance_all)



fileName = "./data/DSL-StrongPasswordData.xls"
#keystroke_data = make_estimate_data_for_up('s002',0,2,fileName)
keystroke_data = make_estimate_data_for_up_all(fileName,'s002')


print(keystroke_data)
print(len(keystroke_data))





