import cv2
import numpy as np
import math
import os


# path = 'templates'
# orb = cv2.ORB_create(nfeatures=1000)

# images = []
# classNames = []
# listaTmp = os.listdir(path)

# for cl in listaTmp:
#     imgCur = cv2.imread(f'{path}/{cl}',0)
#     _, imgCur = cv2.threshold(imgCur,120,255,cv2.THRESH_BINARY)
#     images.append(imgCur)
#     classNames.append(os.path.splitext(cl)[0])



# def findDes(images):
#     desList = []
#     for img in images:
#         kp, des = orb.detectAndCompute(img,None)
#         desList.append(des)
#     return desList

# def findID(img, desList):
#     kp2, des2 = orb.detectAndCompute(img,None)
#     bf = cv2.BFMatcher()
#     matchList = []
#     finVal = -1
#     thres = 2
#     for des in desList:
#         matches = bf.knnMatch(des,des2,k=2)
#         good = []
#         for m,n in matches:
#             if m.distance < 0.75 * n.distance:
#                 good.append([m])
#         matchList.append(len(good))

#     if len(matchList)!= 0:
#         if max(matchList) > thres:
#             finVal = matchList.index(max(matchList))
#     return finVal

# ------------------------------------------------------------------------------------------

video = cv2.VideoCapture("znaki_vid.mp4")
tmp = cv2.imread("znaczek50.jpg",0)
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
    #Trojkat
    trojkat1 = [(w/6, h - h/4.5),(w / 2.4, h / 2),(w-w/4, h-h/4.5)]
    mask = np.zeros_like(gray1)
    cv2.fillPoly(mask,np.array([trojkat1], np.int32), 255)
    gray1 = cv2.bitwise_and(gray1, mask)

    img_thr2 = cv2.adaptiveThreshold(gray1,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,71,-20)

    # Proba wykrywania znaku za pomoca cech ORB

    # trojkat2 = [(w/3, h - h/4),(w / 2.4, h / 2),(w-w/2.5, h-h/4)]
    # mask2 = np.zeros_like(gray1)
    # cv2.fillPoly(mask2,np.array([trojkat2], np.int32), 255)
    # img_thr2 = cv2.bitwise_and(img_thr2, mask2)
    # desList = findDes(images)
    # id = findID(img_thr2,desList)

    # if id != -1:
    #     cv2.putText(orig_frame,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)



    #Rozpoznawanie znaków na drodze

    _, progt = cv2.threshold(tmp,120,255,cv2.THRESH_BINARY)
    _, progs = cv2.threshold(gray1,135,255,cv2.THRESH_BINARY)

    match = cv2.matchTemplate(progs, progt, cv2.TM_CCOEFF_NORMED)
    # 0.32 dziala spoko
    loc = np.where(match >= 0.35)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(orig_frame, pt, (pt[0] + szerokosc, pt[1] + wysokosc), (0,0,255), 3)
        cv2.putText(orig_frame,"Znaczek 50 na godz",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

    _, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

    thresh = cv2.adaptiveThreshold(gray,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,71,-20)

    edges = cv2.Canny(gray, 50, 180)
    
    
    #Prostokat
    
    # maska = np.zeros(edges.shape[:2], np.uint8)
    # maska[380:h-150, 300:w-300] = 255
    # edges[maska!=255] = 0
    h,w = edges.shape[:2]
    #Trojkat
    trojkat = [(w/6, h - h/4.5),(w / 2.4, h / 2),(w-w/4, h-h/4.5)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask,np.array([trojkat], np.int32), 255)
    edges = cv2.bitwise_and(edges, mask)
    

    # Detekcja linii bocznych ulicy
    # lines = cv2.HoughLinesP(edges,
    #                 rho = 2,
    #                 theta = np.pi/180,
    #                 threshold = 100,
    #                 minLineLength = 40,
    #                 maxLineGap = 110)
    lines = cv2.HoughLinesP(edges,
                    2,
                    np.pi/180,
                    170,
                    np.array([]),
                    minLineLength = 70,
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

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) > 0.4:
                cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    #v = np.hstack((gray,edges))
    cv2.imshow("frame", orig_frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()





