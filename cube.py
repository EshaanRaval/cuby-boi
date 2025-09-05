import cv2 as cv
import numpy as np


def detectColour(frame, coords: list) -> str:
    """input frame and the coords for this function to determine the colour at the particular coord"""
    # Blue Mask
    bluel = np.array([94, 80, 2], np.uint8) # today is a bad day to be wearing blue huh? (im eyeballin my colours for my own cube)
    blueu = np.array([120, 255, 255], np.uint8) ### IT DOESN"T MAKE SENSE ???? why does this work
    bmask = cv.inRange(hsv, bluel, blueu)
    blue = cv.bitwise_and(frame, frame, mask = bmask) # could use this line for multiple colours??
    
    # Green Mask
    greenl = np.array([50, 100, 10], np.uint8) 
    greenu = np.array([70, 255, 255], np.uint8) 

    gmask = cv.inRange(hsv, greenl, greenu)
    green = cv.bitwise_and(frame, frame, mask = gmask)

    # Red Mask (works perfectly)

    redl = np.array([169,165,100])
    redu = np.array([189,255,255])
    rmask = cv.inRange(hsv, redl, redu)
    red = cv.bitwise_and(frame, frame, mask = rmask)
    
    # yellow (works perfectly)
    yellowl = np.array([15,100,100])
    yellowu = np.array([35, 255, 255])
    ymask = cv.inRange(hsv, yellowl, yellowu)
    yellow = cv.bitwise_and(frame, frame, mask = ymask)

    # orange # works but is a bit dubious might need to enforce cube corner detection (will work in controlled environment)
    orangel = np.array([2,100,135])
    orangeu = np.array([15,255,255])
    omask = cv.inRange(hsv, orangel, orangeu)
    orange = cv.bitwise_and(frame, frame, mask = omask)
    
    # white (works perfectly)
    whitel = np.array([0,0,235])
    whiteu = np.array([255,20,255])
    wmask = cv.inRange(hsv, whitel, whiteu)
    white = cv.bitwise_and(frame, frame, mask = wmask)

    colours = {"white": white,"yellow": yellow,"orange": orange,"green": green, "red": red, "blue": blue}
    detected = []
    #print(coords)
    for colour in colours.values():
        pixel = colour[round(int(coords[0])), round(int(coords[1]))]
        # print(pixel)
        if np.all(pixel != [0,0,0]):
            detected.append(colours.keys(colour))
    if detected:
        # print(detected)
        c2rgb = {"white": [255,255,255], "yellow": [255, 255, 0], "orange": [255, 165, 0], "green": [0,255,0], "red": [255,0,0], "blue": [0,0,255]}
        return c2rgb[detected[0]]
    else:
        return pixel


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    sframe = cv.GaussianBlur(frame, (5, 5), 0)
    mframe = cv.medianBlur(frame, 5)
    

    
    
    
    # now detect a big cube 
    
    # so here's the big idea we detect squares within square (cubeparts within cubeface)
    # then we try to detect the colour at the very center of the shape (and try to approximate the closest colour to the six we have)
    # then we just store that colour in a colour matrix 3x3?
    
    # Detect the cubeface: RETR_EXTERNAL then do some calculation to find the cubeparts then run the colour detection and then colour matrix?
    
    # Detecting cubeface
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # don't use any other frames as they blur the edge too much for detection
    decreasenoise = cv.fastNlMeansDenoising(gray, 10,15,7,21)
    blurred = cv.GaussianBlur(decreasenoise, (3,3), 0)
    canny = cv.Canny(blurred, 20, 40)
    thresh = cv.threshold(canny, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)[1]
    contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) ==2 else contours[1]
    for i, contour in enumerate(contours):
        if i == 0:
            continue
        x,y,w,h = cv.boundingRect(contour)
        if cv.contourArea(contour)> 1000:        
            
            xmid = x+ w/2
            ymid = y + h/2
            cv.rectangle(frame, (x,y), (x+w, y+h), [0,0,0], 2)
        
    # edges = cv.Canny(gray, 100, 255) # extremely useful for detecting cubeface and cubeparts
    # _, tframe = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)
    # contours, hierarchy = cv.findContours(tframe, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # what's the diff between RETR tree and external?
    



    # params to detect the cubeparts and face: cnt area >1000  <1500, iscontourconvex and obviously 



    ######## Dubious Work
    
    
    
    
    # data = []
    # for contour in contours:
    #     epsilon = 0.1*cv.arcLength(contour,True)
    #     approx = cv.approxPolyDP(contour, epsilon, True)
    #     if len(approx) == 4:
    #         data.append(cv.contourArea(contour))
    #     else:
    #         data.append(0)
    
    # bcontour = max(data)
    # index = data.index(bcontour)
    # for i, contour in enumerate(contours):
    #     if i == 0:
    #         continue # because the first shape is going to be the shape/dimension of the video capture
        
    #     epsilon = 0.1*cv.arcLength(contour,True)
    #     approx = cv.approxPolyDP(contour, epsilon, True)
    #     if i == index:
    #         x,y,w,h = cv.boundingRect(approx)
    #         rFrame = cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)

        # if len(approx) == 4 and cv.contourArea(contour) in range(10,100):
        #     x,y,w,h = cv.boundingRect(approx)
        #     xmid = x + w/2  # coz square innit
        #     ymid = y + h/2
        #     coords = (int(xmid), int(ymid))
        #     cv.drawContours(frame, contour, 0, (255,255,255), 4)
        #     rFrame = cv.rectangle(frame, (x, y),  (x + w, y + h),  (0, 0, 255), 2)
        #     font = cv.FONT_HERSHEY_SIMPLEX
        #     cv.putText(frame, "Cubepart/face??", coords, font, 1, (255,255,255), 2)
        
        




    # cubeface = cv.goodFeaturesToTrack(gray, 1, 0.75, 10)
    # cubeface = np.intp(cubeface)
    # for corner in cubeface:
    #     x,y = corner.ravel()
    #     cv.circle(red, (x,y), 5, (255,0,0), -1)

    # cubeparts = cv.goodFeaturesToTrack(frame, 28,1)
    

    cv.imshow('frame',canny)


    if cv.waitKey(1) == ord('x'):
        break




cap.release()
cv.destroyAllWindows()
