# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:04:11 2017

@author: Plenoi
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split

# SVM
from sklearn.model_selection import GridSearchCV 
from sklearn import svm

import numpy as np
from pandas import read_csv, read_excel,pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

lines = list(read_csv('thaiword.txt',header=None).iloc[:,0])

def isThai(chr):
    cVal = ord(chr)
    if(cVal >= 3584 and cVal <= 3711):
        return True
    return False

def scappyKNN():
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

def create_model():
	# create model
    model = Sequential()
    model.add(Dense(10, input_dim=4225, activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(6,activation='relu'))
    model.add(Dense(2, activation='softmax'))
	
    # Compile mode
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def main():
    # อ่าน Comment จาก CSV file โดยใช้ , เป็นตัวแบ่ง column โดยเอา column แรก [:,0]
    # ออกมา เนื่องจากไฟล์เรา 1 บรรทัดแทน 1 ข้อความ
    comments_data = read_excel('data.xlsx',header=None)
    
    comments = np.array(comments_data.iloc[:,0])
    label = np.array(comments_data.iloc[:,1],dtype=int)

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
    
    # word_list = ('เพลิน','หลับ','ดี','ชอบ','เวอร์ชั่น','เฉย','สนุก','แย่','พลาด')
    # สร้าง matrix (ตาราง) ชื่อ data เพื่อนับว่าในแต่ละข้อความมีคำที่เกิดขึ้นใน wordlist กี่คำ
    # แถว (row) แต่ละแถวคือข้อความแต่ละอัน คอลัมภ์ แต่ละคอลัมภ์เก็บค่าความถี่แต่ละคำ
    data = np.zeros((len(comments_cut),len(word_list))) 
    for i in range(0,len(comments_cut)):
        for j in range(0,len(word_list)):
            data[i,j] = comments_cut[i].count(word_list[j])
            
            
     # Train Test Split
    [X_train, X_test, y_train, y_test] = train_test_split(data,
                                                        label,
                                                        test_size = 0.3,
                                                        random_state=1)
    
    # Keras Deep
    '''
    clf = KerasClassifier(build_fn=create_model, epochs=300, batch_size=10)
    '''
    
    # SVM
    '''
    parameters = {'C':[1,2,4,8,16,32],
                  #'gamma':[0.001,0.05,0.01,0.5,0.1,0,1,2,4,8,16,32]}
    #clf = GridSearchCV(svm.SVC(),parameters,cv=10,scoring='accuracy')   
    #clf.best_score_*100
    '''
    
    
    #Decision Tree
    '''
    clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
    '''
    
    
    #KNN
    '''
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    '''
    
    prediction = clf.predict(X_test)
    print(accuracy_score(y_test,prediction))

    
            
            

if __name__ == "__main__":
    main()