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
def make_estimate_data_for_down(user,line_start,line_end,fileName):
    df_oral = pd.read_excel(fileName,header=0)
    df = df_oral[df_oral["subject"]==user]
    #df = pd.read_excel(fileName, header=0)
    data_instance = []  # 一次击键行为对应的数据。
    data_instance_all = []
    data_temp = []  # 临时list变量。
    temp_DD = 0.0  # Down-Down临时变量
    temp_Hole = 0.0  # Hold临时编码
    sum_time = 0.0  # 当前时间

    for i in range(line_start, line_end):
        for j in range(11):
            if j > 0:
                temp_DD = df.iloc[i, 1 + j * 3]  # down时间。
                sum_time += temp_DD  # down事件。
                #print(sum_time)
                data_temp.append(sum_time)  # down事件加入list。
                #print(data_temp)
            temp_Hole = df.iloc[i, 3 + j * 3]  # Hold的时间。
            sum_time += temp_Hole  # up事件。
            #data_temp.append(sum_time)  # up事件加入list。
        data_instance.append(data_temp)
        data_instance_all.append(data_instance)
        data_temp = []
        data_instance = []
        sum_time = 0.0

    return np.concatenate(data_instance_all)

def make_estimate_data_for_down_all(fileName,user):
    df = pd.read_excel(fileName,header=0)
    #df = df_oral[df_oral["subject"]==user]
    #df = pd.read_excel(fileName, header=0)
    data_instance = []  # 一次击键行为对应的数据。
    data_instance_all = []
    data_temp = []  # 临时list变量。
    temp_DD = 0.0  # Down-Down临时变量
    temp_Hole = 0.0  # Hold临时编码
    sum_time = 0.0  # 当前时间

    for i in range(0, 20400):
        #跳过user
        if df.iloc[i,0] == user:
            continue
        # 只取每个用户的前50个样本
        if i%400 >= 50:
            continue
        for j in range(11):
            if j > 0:
                temp_DD = df.iloc[i, 1 + j * 3]  # down时间。
                sum_time += temp_DD  # down事件。
                #print(sum_time)
                data_temp.append(sum_time)  # down事件加入list。
                #print(data_temp)
            temp_Hole = df.iloc[i, 3 + j * 3]  # Hold的时间。
            sum_time += temp_Hole  # up事件。
            #data_temp.append(sum_time)  # up事件加入list。
        data_instance.append(data_temp)
        data_instance_all.append(data_instance)
        data_temp = []
        data_instance = []
        sum_time = 0.0

    return np.concatenate(data_instance_all)




'''
fileName = "./data/DSL-StrongPasswordData.xls"
#keystroke_data = make_estimate_data('s032',10400,10402,fileName)
keystroke_data = make_estimate_data_for_down_all(fileName,'s002')

print (keystroke_data)
print (len(keystroke_data))
#print (len(keystroke_data))
'''