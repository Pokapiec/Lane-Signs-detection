import cv2
import numpy as np



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

match = cv2.matchTemplate(progs, progt, cv2.TM_CCOEFF_NORMED)
loc = np.where(match >= 0.32)
# print(loc)
# print("----------------")
# print(list(zip(loc)))
# print("----------------")
# print(list(zip(*loc)))
for pt in zip(*loc[::-1]):
    cv2.rectangle(img2, pt, (pt[0] + szerokosc, pt[1] + wysokosc), (0,0,255), 3)

#bottom_right = (location[0] + szerokosc, location[1] + wysokosc)
#cv2.rectangle(img,location,bottom_right,(0,0,255),5)
_, prog1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

prog = cv2.adaptiveThreshold(gray,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,71,-30)

#cv2.morphologyEx(prog,cv2.MORPH_OPEN,np.ones((13,13),np.uint8))
kontury = cv2.Canny(prog1,50,180)
# lines = cv2.HoughLinesP(kontury,
#                     rho = 1,
#                     theta = np.pi/180,
#                     threshold = 170,
#                     minLineLength = 70,
#                     maxLineGap = 100)
lines = cv2.HoughLinesP(kontury,
                    2,
                    np.pi/180,
                    170,
                    np.array([]),
                    minLineLength = 70,
                    maxLineGap = 100)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)


cv2.circle(img, (620,500),5,(0,0,255),-1)
cv2.circle(img, (790,500),5,(0,0,255),-1)

cv2.circle(img, (250,700),5,(0,0,255),-1)
cv2.circle(img, (1150,710),5,(0,0,255),-1)

pts1 = np.float32([[620,500],[790,500],[250,700],[1150,710]])
pts2 = np.float32([[0,0],[600,0],[0,600],[600,600]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)

result = cv2.warpPerspective(img3,matrix,(600,600))


#print(match)
#cv2.imshow("wsf",prog)
#cv2.imshow("wsf1",kontury)
cv2.imshow("img1",img)
cv2.imshow("img",result)

#cv2.imshow("img",match)
cv2.waitKey(0)
cv2.destroyAllWindows()
