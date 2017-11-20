# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:04:11 2017

@author: Plenoi
"""
import numpy as np
from pandas import read_excel 
from pandas import read_csv

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

lines = list(read_csv('thaiword.txt',header=None).iloc[:,0])

def isThai(chr):
    cVal = ord(chr)
    if(cVal >= 3584 and cVal <= 3711):
        return True
    return False
    
def tokenize(string, length):
    string = ''.join(list(filter(isThai, string)))
    string = string.replace(' ','')
    lines.sort(key=len, reverse=True)
    result = list()
    for word in lines:
        if word in string and len(word) > length:
            result.append(word)
            string = string.replace(word,'',1)
    return result

def main():
    # อ่าน Comment จาก CSV file โดยใช้ , เป็นตัวแบ่ง column โดยเอา column แรก [:,0]
    # ออกมา เนื่องจากไฟล์เรา 1 บรรทัดแทน 1 ข้อความ
    comments = read_excel('data.xlsx',header=None)

    # ตัดคำของแต่ละข้อความในแต่ละบรรทัดมาเก็บไว้ที่ไฟล์ชื่อ comments_cut
    # สร้างคำทั้งหมดที่มีจากข้อความของเราเอาไว้ใน word_list
    comments_cut = list()
    word_list = list()
    for comment_lines in comments:
        tmp = tokenize(comment_lines,2) # จำนวนคำที่สั้นที่สุด 2 ตัวอักษร ที่จะตัด
        comments_cut.append(' '.join(tmp))
        word_list = word_list + tmp
    word_list = np.unique(word_list)
    comments_cut = np.array(comments_cut)
    
    # สร้าง matrix (ตาราง) ชื่อ data เพื่อนับว่าในแต่ละข้อความมีคำที่เกิดขึ้นใน wordlist กี่คำ
    # แถว (row) แต่ละแถวคือข้อความแต่ละอัน คอลัมภ์ แต่ละคอลัมภ์เก็บค่าความถี่แต่ละคำ
    data = np.zeros((len(comments_cut),len(word_list))) 
    for i in range(0,len(comments_cut)):
        for j in range(0,len(word_list)):
            data[i,j] = comments_cut[i].count(word_list[j])
            
            
            

if __name__ == "__main__":
    main()