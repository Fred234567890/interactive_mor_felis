# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:06:10 2022

@author: quetscher
"""
import os
import csv
import pandas
import time

    
def csv_writeLine(filename,data,lineIndex):
    if os.path.isfile(filename+'.csv'):
        df = pandas.read_csv(filename+'.csv',header=None)
        df.loc[lineIndex]=data
        df.to_csv(filename+'.csv',index=False,header=False)
    else:
        with open(filename+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([data])

def csv_readLine(filename,lineIndex):
    df = pandas.read_csv(filename+'.csv',header=None)
    return df.loc[lineIndex].tolist()

def timeprint(string):
    print(time.strftime("%H:%M:%S", time.gmtime()) + ': ' + str(string))