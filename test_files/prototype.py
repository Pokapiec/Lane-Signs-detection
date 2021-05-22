import cv2
import numpy as np
import os


path = 'templates'
orb = cv2.ORB_create(nfeatures=1000)

images = []
classNames = []
listaTmp = os.listdir(path)

for cl in listaTmp:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    _, imgCur = cv2.threshold(imgCur,120,255,cv2.THRESH_BINARY)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])



def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finVal = -1
    thres = 2
    for des in desList:
        matches = bf.knnMatch(des,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        matchList.append(len(good))

    if len(matchList)!= 0:
        if max(matchList) > thres:
            finVal = matchList.index(max(matchList))
    return finVal


desList = findDes(images)

img6 = cv2.imread("znak2.jpg")
imgorig = img6.copy()


img6 = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY)

wys, szer = img6.shape[:2]

maska = np.zeros(img6.shape[:2], np.uint8)
maska[490:wys-200, 200:szer-400] = 255

img6[maska!=255] = 0

_, img6_th = cv2.threshold(img6,120,255,cv2.THRESH_BINARY)
id = findID(img6_th,desList)

if id != -1:
    cv2.putText(imgorig,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA) 


cv2.imshow("orig", imgorig)
cv2.waitKey(0)






























# Detekcja wcześniej zdefiniowanych znaków naziemnych metoda matchTemplates
img = cv2.imread("znak2.jpg")
tmp = cv2.imread("znaczek50.jpg")

tmp_gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

wys, szer = img.shape[:2]

wysokosc, szerokosc = tmp.shape[:2]
img2 = img.copy()
img3 = img.copy()



#blured = cv2.GaussianBlur(img2, (5, 5), 0)

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
maska = np.zeros(img2.shape[:2], np.uint8)
maska[490:wys-200, 200:szer-400] = 255

gray[maska!=255] = 0
#min, max, min_loc, max_loc = cv2.minMaxLoc(match)
#location = max_loc

_, progt = cv2.threshold(tmp_gray,120,255,cv2.THRESH_BINARY)
_, progs = cv2.threshold(gray,135,255,cv2.THRESH_BINARY)


# orb = cv2.ORB_create(nfeatures = 1000)

# kp1, des1 = orb.detectAndCompute(progs,None)
# kp2, des2 = orb.detectAndCompute(progt,None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)

# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# img4 = cv2.drawMatchesKnn(img2,kp1,tmp,kp2,good,None)


# match = cv2.matchTemplate(progs, progt, cv2.TM_CCOEFF_NORMED)
# loc = np.where(match >= 0.32)
# # print(loc)
# # print("----------------")
# # print(list(zip(loc)))
# # print("----------------")
# # print(list(zip(*loc)))
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img2, pt, (pt[0] + szerokosc, pt[1] + wysokosc), (0,0,255), 3)

#bottom_right = (location[0] + szerokosc, location[1] + wysokosc)
#cv2.rectangle(img,location,bottom_right,(0,0,255),5)
_, prog1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

prog = cv2.adaptiveThreshold(gray,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,71,-30)

#cv2.morphologyEx(prog,cv2.MORPH_OPEN,np.ones((13,13),np.uint8))
kontury = cv2.Canny(prog1,50,180)


# Tutaj sobie robiłem perspektywe w nadzieji o lepsze jutro
# cv2.circle(img, (620,500),5,(0,0,255),-1)
# cv2.circle(img, (790,500),5,(0,0,255),-1)

# cv2.circle(img, (250,700),5,(0,0,255),-1)
# cv2.circle(img, (1150,710),5,(0,0,255),-1)

# pts1 = np.float32([[620,500],[790,500],[250,700],[1150,710]])
# pts2 = np.float32([[0,0],[600,0],[0,600],[600,600]])

# matrix = cv2.getPerspectiveTransform(pts1,pts2)

# result = cv2.warpPerspective(img3,matrix,(600,600))

# result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# _, progP = cv2.threshold(result_gray,120,255,cv2.THRESH_BINARY)
# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(progP, cv2.MORPH_ERODE, kernel)
# edges_persp = cv2.Canny(opening,50,180)


#cv2.imshow("wsf",prog)
#cv2.imshow("wsf1",kontury)
# cv2.imshow("img1",img)

# cv2.imshow("img",img4)

cv2.waitKey(0)
cv2.destroyAllWindows()
