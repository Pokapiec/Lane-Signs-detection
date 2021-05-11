# Algorytm SURF, chyba jest platny i licencja kosztuje xD

# img = cv2.imread("znak1.jpg",0)

# surf = cv2.xfeatures2d.SIFT_create()

# kp, des = surf.detectAndCompute(img,None)
# img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
# cv2.imshow("a",img2)

# # Okrągle znaki na ulicach ???
# circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=0,maxRadius=0)
# circles = np.uint16(np.around(circles))

# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(edges,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(edges,(i[0],i[1]),2,(0,0,255),3)
import cv2
import numpy as np
import math

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
    region_of_interest_vertices = [(w/6, h - h/4.5),(w / 2.4, h / 2),(w-w/4, h-h/4.5)]
    mask = np.zeros_like(gray1)
    cv2.fillPoly(mask,np.array([region_of_interest_vertices], np.int32), 255)
    gray1 = cv2.bitwise_and(gray1, mask)

    # Rozpoznawanie znaków na drodze

    _, progt = cv2.threshold(tmp,120,255,cv2.THRESH_BINARY)
    _, progs = cv2.threshold(gray1,135,255,cv2.THRESH_BINARY)

    match = cv2.matchTemplate(progs, progt, cv2.TM_CCOEFF_NORMED)
    # 0.32 dziala spoko
    loc = np.where(match >= 0.35)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(orig_frame, pt, (pt[0] + szerokosc, pt[1] + wysokosc), (0,0,255), 3)

    # _, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

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


    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) > 0.4:
                cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.imshow("frame", orig_frame)
    cv2.imshow("edges", gray1)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()





