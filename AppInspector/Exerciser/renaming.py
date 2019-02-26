import os
import re

user = 'hfu'
apk_dir = '/home/hao/PycharmProjects/TrafficAnalysis/data_collection/output/CTU-13-1/0' #'C:\Users\\' + user + '\Documents\FlowIntent\\apks\\drebin\\SerBG\\'
for root, dirs, files in os.walk(apk_dir, topdown=False):
    for filename in files:
        if re.search('json$', filename) and ':' in filename:
            newfilename = filename.replace(':', '_')
            os.rename(os.path.join(root, filename), os.path.join(root, newfilename))
