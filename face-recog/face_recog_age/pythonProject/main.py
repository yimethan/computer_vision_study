import cv2
import numpy as np
import face_recognition

# 사진 가져오기
imgElon = face_recognition.load_image_file('ImagesBasic/ElonMusk.jpg')
# 라이브러리에서는 이미지를 RGB로 이해하는데 우리는 이미지를 BGR로 가져오니까 변환해 줘야 함
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
# test 이미지 마찬가지로
imgTest = face_recognition.load_image_file('ImagesBasic/ElonTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# 얼굴 인식하기 위해서 얼굴 위치를 지정함. 이미지는 하나 뿐이니 첫번째 요소를 faceLoc 변수에 저장해줌
faceLoc = face_recognition.face_locations(imgElon)[0]
# 그리고 이걸 인코딩시킬 거임
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3],faceLoc[0]), (faceLoc[1], faceLoc[2]),(255, 0, 0),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1], faceLocTest[2]),(255, 0, 0),2)

# 비교하는 얼굴 간 유사도
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('ElonMusk', imgElon)
cv2.imshow('ElonTest', imgTest)
cv2.waitKey(0)
