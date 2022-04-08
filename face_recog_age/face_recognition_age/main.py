import face_recognition
import cv2
import os
import numpy as np


def getImages(path):
    images = []
    files = os.listdir(path)
    files = [dir for dir in files if not dir.startswith('.')]
    for i in files:
        curImg = cv2.imread(f'{path}/{i}')
        images.append(curImg)
        # print(i)
    return images

def predictAge(ageRangeList):
    predictList = []  # 가장 유사한 이미지의 이름
    for i in ageRangeList:
        imgSmall = cv2.resize(i, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
        try:
            encodeCurFrame = face_recognition.face_encodings(imgSmall)[0]
            faceDis = face_recognition.face_distance(knownList, encodeCurFrame)  # 다른 정도 (knownList, 즉 인코딩된 순서, 데이터세트 순)
            matchIdx = np.argmin(faceDis)  # 차이가 가장 적은 얼굴의 인덱스
            matchName = knownNames[matchIdx]
            predictList.append(matchName)
        except IndexError:
            pass
    # print(predictList)
    return predictList


age_list = ['kids', 'youth', 'senior']


# 데이터세트 파일 경로
path = 'faces'
files = [dir for dir in os.listdir(path) if not dir.startswith('.')]
knownNames = []
knownList = []
for i in files:
    curImg = cv2.imread(f'{path}/{i}')
    encode = face_recognition.face_encodings(curImg)[0]
    knownList.append(encode)
    knownNames.append(os.path.splitext(i)[0].split('.')[0])
    # print(i)


for i in age_list:
    testImg = getImages(f'{i}Test')
    agePredictList = predictAge(testImg)
    print('{} Accuracy: {}%' .format(i, agePredictList.count(i)*100/len(testImg)))
