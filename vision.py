import cv2
import numpy as np
import math
import os


# path = 'templates'
# orb = cv2.ORB_create(nfeatures=1000)
#
# images = []
# classNames = []
# listaTmp = os.listdir(path)
#
# for cl in listaTmp:
#     imgCur = cv2.imread(f'{path}/{cl}',0)
#     _, imgCur = cv2.threshold(imgCur,120,255,cv2.THRESH_BINARY)
#     images.append(imgCur)
#     classNames.append(os.path.splitext(cl)[0])
#
#
#
# def findDes(images):
#     desList = []
#     for img in images:
#         kp, des = orb.detectAndCompute(img,None)
#         desList.append(des)
#     return desList
#
# def findID(img, desList):
#     kp2, des2 = orb.detectAndCompute(img,None)
#     bf = cv2.BFMatcher()
#     matchList = []
#     finVal = -1
#     thres = 3
#     for des in desList:
#         matches = bf.knnMatch(des,des2,k=2)
#         good = []
#         for m,n in matches:
#             if m.distance < 0.75 * n.distance:
#                 good.append([m])
#         matchList.append(len(good))
#
#     if len(matchList)!= 0:
#         if max(matchList) > thres:
#             finVal = matchList.index(max(matchList))
#     return finVal

# ------------------------------------------------------------------------------------------

video = cv2.VideoCapture("duzo_znakow.mp4")
# video = cv2.VideoCapture("cut_vid.mp4")
tmp = cv2.imread("znaczek40.jpg",0)
wysokosc, szerokosc = tmp.shape[:2]



while True:
    ret, orig_frame1 = video.read()
    if not ret:
        break
    
    orig_frame = orig_frame1.copy()

    blured = cv2.GaussianBlur(orig_frame, (7, 7), 0)
    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    h,w = gray1.shape[:2]
    #Trojkat obejmujacy tylko droge
    trojkat1 = [(w/6, h - h/4.5),(w / 2.4, h / 2),(w-w/4, h-h/4.5)]
    mask = np.zeros_like(gray1)
    cv2.fillPoly(mask,np.array([trojkat1], np.int32), 255)
    gray1 = cv2.bitwise_and(gray1, mask)

    # img_thr2 = cv2.adaptiveThreshold(gray1,255,
    #         cv2.ADAPTIVE_THRESH_MEAN_C,
    #         cv2.THRESH_BINARY,71,-20)
    #
    # # Proba wykrywania znaku za pomoca cech ORB
    #
    # trojkat2 = [(w/3, h - h/4),(w / 2.4, h / 2),(w-w/2.5, h-h/4)]
    # mask2 = np.zeros_like(gray1)
    # cv2.fillPoly(mask2,np.array([trojkat2], np.int32), 255)
    # gray11 = cv2.bitwise_and(gray1, mask2)
    # desList = findDes(images)
    # id = findID(gray11,desList)
    #
    # if id != -1:
    #     cv2.putText(orig_frame,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

    # Rozpoznawanie znak??w na drodze

    # progt = cv2.adaptiveThreshold(tmp,255,
    #         cv2.ADAPTIVE_THRESH_MEAN_C,
    #         cv2.THRESH_BINARY,71,-20)
    # progs = cv2.adaptiveThreshold(gray1, 255,
    #           cv2.ADAPTIVE_THRESH_MEAN_C,
    #           cv2.THRESH_BINARY, 71, -20)

    # progt = cv2.Canny(tmp, 50, 180)
    # progs = cv2.Canny(gray1, 50, 180)
    trojkat2 = [(w / 3.4, h - h / 4), (w / 2.4, h / 1.9), (w - w / 2.5, h - h / 4)]
    mask2 = np.zeros_like(gray1)
    cv2.fillPoly(mask2, np.array([trojkat2], np.int32), 255)


    # Canny(50,180) i pr??g 0.14 dzia??a ok

    progs = cv2.Canny(gray, 30, 180)
    progt = cv2.Canny(tmp, 50, 180)
    progs = cv2.bitwise_and(progs, mask2)
    # kernel = np.ones((3,3), np.uint8)
    # _, progt = cv2.threshold(tmp,120,255,cv2.THRESH_BINARY)
    # _, progs = cv2.threshold(gray2,135,255,cv2.THRESH_BINARY)
    # progs = cv2.dilate(progs,kernel)
    # progt = cv2.morphologyEx(progt, cv2.MORPH_CLOSE, kernel)
    # progt = cv2.erode(progt,kernel,iterations=2)

    match = cv2.matchTemplate(progs, progt, cv2.TM_CCOEFF_NORMED)
    # print(match)
    # 0.14 dziala spoko dla cannego
    #
    znak = 0
    loc = np.where(match >= 0.135)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(orig_frame, pt, (pt[0] + szerokosc, pt[1] + wysokosc), (0, 0, 255), 3)
        cv2.putText(orig_frame, "Znaczek 40 mph", (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        znak = 1
        break


    # _, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

    # thresh = cv2.adaptiveThreshold(gray,255,
    #         cv2.ADAPTIVE_THRESH_MEAN_C,
    #         cv2.THRESH_BINARY,71,-20)

    #Prostokat maskuj??cy

    # maska = np.zeros(edges.shape[:2], np.uint8)
    # maska[380:h-150, 300:w-300] = 255
    # edges[maska!=255] = 0
    edges = cv2.Canny(gray, 50, 180)

    h,w = edges.shape[:2]
    #Trojkat
    trojkat = [(w/6, h - h/4.5),(w / 2.4, h / 2),(w-w/4, h-h/4.5)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask,np.array([trojkat], np.int32), 255)
    edges = cv2.bitwise_and(edges, mask)



    # Detekcja linii bocznych ulicy

    lines = cv2.HoughLinesP(edges,
                    1,
                    np.pi/90,
                    40,
                    np.array([]),
                    minLineLength = 40,
                    maxLineGap = 40)



    # cv2.circle(orig_frame, (440,380),5,(0,0,255),-1)
    # cv2.circle(orig_frame, (610,380),5,(0,0,255),-1)

    # cv2.circle(orig_frame, (250,500),5,(0,0,255),-1)
    # cv2.circle(orig_frame, (900,500),5,(0,0,255),-1)

    # pts1 = np.float32([[440,380],[610,380],[250,500],[900,500]])
    # pts2 = np.float32([[0,0],[600,0],[0,600],[600,600]])
    # matrix = cv2.getPerspectiveTransform(pts1,pts2)
    # result = cv2.warpPerspective(orig_frame,matrix,(600,600))

    # result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # _, progP = cv2.threshold(result_gray,120,255,cv2.THRESH_BINARY)
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(progP, cv2.MORPH_ERODE, kernel)
    # edges_persp = cv2.Canny(opening,50,180)
    slopeLeft = []
    slopeRight = []
    intLeft = []
    intRight = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # slope = (y2 - y1) / (x2 - x1)
            prosta = np.polyfit((x1, x2), (y1, y2), 1)
            if math.fabs(prosta[0]) > 0.2:
                if (x1 > w/2.7 and x2 > w/2.7):
                    slopeRight.append(prosta[0])
                    intRight.append(prosta[1])
                    # cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

                elif (x1 < w/2.7 and x2 < w/2.7):
                    slopeLeft.append(prosta[0])
                    intLeft.append(prosta[1])
                    # cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if len(slopeLeft) < 1 or len(slopeRight) < 1:
        cv2.putText(orig_frame, "Nie wykryto zadnych krawedzi !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 1,
                    cv2.LINE_AA)
    elif len(slopeLeft) >= 1 and len(slopeRight) >= 1:
        # Linia prawa
        y1 = np.int32(h - h / 5)
        y2 = np.int32(y1 * 7 / 10)
        x1 = np.int32(w - w / 2.5)
        x2 = np.int32((y2 - np.average(intRight)) / np.average(slopeRight) )
        cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Linia lewa
        y1 = np.int32(h - h / 5)
        y2 = np.int32(y1 * 7 / 10)
        x1 = np.int32(w / 5)
        x2 = np.int32((y2 - np.average(intLeft)) / np.average(slopeLeft))
        cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    #v = np.hstack((gray,edges))
    cv2.imshow("frame", orig_frame)
    if znak == 1:
        cv2.waitKey(4000)
    cv2.imshow("edges", gray)

    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
