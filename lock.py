import cv2
import numpy as np
import sys

secret_key = ["center center splay", "upper right fist", "center center fist"]
correct = 0

for i in range(1,len(sys.argv)):
    #read image from file
    img = cv2.imread(sys.argv[i])

    #reduce the image size for easier processing
    scale_percent = 25
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #blur image to generalize shapes
    blur_factor = (30, 30)
    blurred_img = cv2.blur(img, blur_factor) 

    #conver from BGR to HSV to filter out the green background
    hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    #mask green from image
    mask = cv2.inRange(hsv, (36, 25, 25), (100, 255,255))

    #convert to binary image
    ret,thresh1 = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    cv2.imshow('thresh1',thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #find contours in image
    contours, hierarchy = cv2.findContours(image=thresh1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)


    #take second contour, ignore outisde image contour
    if (len(contours) < 2):
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        gesture = "unknown"
    else:
        contour = contours[1]
        
        '''the following code for calculating the angle of the defects using the law of 
        cosines is developed from the following source which is public to use:
        https://theailearner.com/2020/11/09/convexity-defects-opencv/'''
        
        #find convex hull outline of hand
        convexHull = cv2.convexHull(contour)

        #draw contour on image
        #cv2.drawContours(img, [contour], -1, (255,0,0), 2)
        cv2.drawContours(img, [convexHull], -1, (255, 0, 0), 2)

        #find convex hull but with no return points to pass to defects function
        convexHullnopoints = cv2.convexHull(contour, returnPoints=False)

        #find defects in hull vs contour to identify number of fingers showing
        defects = cv2.convexityDefects(contour, convexHullnopoints)
        hand = contour

        #find area of convex hull to approximate hand shape size
        convex_hull_area = cv2.contourArea(convexHull)
        #print(str(convex_hull_area))

        #counter variable to store number of fingers visible in image
        cnt = 0
        #check number of defects, if none then no need to count
        if defects is not None:
            #check angle of defect
            for defect in defects:  # calculate the angle
                #pull indices for the defect checking
                s, e, f, d = defect[0]
                start_point = tuple(hand[s][0])
                end_point = tuple(hand[e][0])
                farthest_point = tuple(hand[f][0])
                
                #use cosine theorem to find angle of the defect
                a = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
                b = np.sqrt((farthest_point[0] - start_point[0]) ** 2 + (farthest_point[1] - start_point[1]) ** 2)
                c = np.sqrt((end_point[0] - farthest_point[0]) ** 2 + (end_point[1] - farthest_point[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                
                #if angle is less than 90 degrees (pi/2) then count the defect as a finger
                if angle <= np.pi / 2:
                    cnt += 1
                    cv2.circle(img, farthest_point, 4, [0, 0, 255], -1)
        
        '''end cited code'''

        #analysis of image processing results
        #print(str(cnt))
        #print(str(convex_hull_area))
        if cnt == 4:
            gesture = "splay"
        elif cnt == 0:
            if convex_hull_area > 45000 and convex_hull_area < 100000:
                gesture = "palm"
            elif convex_hull_area > 15000 and convex_hull_area < 45000:
                gesture = "fist"
            else:
                gesture = "unknown"
        elif cnt == 1:
            if convex_hull_area > 40000 and convex_hull_area < 100000:
                gesture = "palm"
            else:
                gesture = "unknown"
        else:
            gesture = "unknown"

    #get height and width of binary image
    h, w = thresh1.shape

    #dict to store number of black pixels in each segment of the image
    pixel_count_dict = {}

    #segment image into nine separate parts
    upper_left = thresh1[:h//3,:w//3]
    upper_center = thresh1[:h//3, w//3:w - w//3]
    upper_right = thresh1[:h//3, w - w//3:w]

    center_left = thresh1[h//3:h - h//3,:w//3]
    center_center = thresh1[h//3:h - h//3, w//3:w - w//3]
    center_right = thresh1[h//3:h - h//3, w - w//3:w]

    lower_left = thresh1[h - h//3:h,:w//3]
    lower_center = thresh1[h - h//3:h, w//3:w - w//3]
    lower_right = thresh1[h - h//3:h, w - w//3:w]

    #count number of black pixels in each segment and add to dict
    pixel_count_dict["upper left "] = np.sum( upper_left == 0 )
    pixel_count_dict["upper center "] = np.sum( upper_center == 0 )
    pixel_count_dict["upper right "] = np.sum( upper_right == 0 )

    pixel_count_dict["center left "] = np.sum( center_left == 0 )
    pixel_count_dict["center center "] = np.sum( center_center == 0 )
    pixel_count_dict["center right "] = np.sum( center_right == 0 )

    pixel_count_dict["lower left "] = np.sum( lower_left == 0 )
    pixel_count_dict["lower center "] = np.sum( lower_center == 0 )
    pixel_count_dict["lower right "] = np.sum( lower_right == 0 )

    #segment of image with the most black pixels is the location of the hand
    position = max(pixel_count_dict, key = pixel_count_dict.get)
    cv2.putText(img, position + gesture, (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
    
    print(str(position + gesture + " detected"))
    if position + gesture == secret_key[i - 1]:
        correct = correct + 1

    # Display the final identified image
    cv2.imshow('Binary', thresh1)
    cv2.imshow('Final', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if correct == 3:
    print("door unlocked")
else:
    print("combination failed")