import sys
sys.path.insert(1, "/usr/local/lib/python3.5/dist-packages/cv2")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import numpy as np
import cv2, os, sys, getopt
import argparse
import datetime 
from math import sqrt,pi,cos,sin,acos,asin
Generic_location = os.path.abspath(__file__ + "/..")
print("Generic_location ", Generic_location)

parser = argparse.ArgumentParser()

parser.add_argument('--TrainDIR', type=str, default='', help='path to Trainingdata')
parser.add_argument('--SaveDIR', type=str, default='Label_txt/', help='path to train.txt')
parser.add_argument('--DefaultLabel', type=str, default='0', help='Number of object label')
FLAGS = parser.parse_args()

print("FLAGS.TrainDIR: ", FLAGS.TrainDIR)

day = str(datetime.datetime.now()).split(" ")[0]
time = str(datetime.datetime.now()).split(" ")[1]
time = time.split(":")[0] + "_" + time.split(":")[1] + "_" + time.split(":")[2].split(".")[0]

current_time = day + "_" + time + "_"

# Global Variables

# Arguments
pathIMG = ''
pathDIR = ''
pathSAV = ''
regex = ''

pathDIR = FLAGS.TrainDIR
pathSAV = FLAGS.SaveDIR
object_label = FLAGS.DefaultLabel

#===============
bboxes = []
rotation_LU = []
rotation_LD = []
rotation_RU = []
rotation_RD = []
rotation_cen = []
p = 0

rotate_5 = np.array([[cos(5*pi/180) , -sin(5*pi/180)],
                    [sin(5*pi/180) , cos(5*pi/180)]])
rotate_2 = np.array([[cos(2*pi/180) , -sin(2*pi/180)],
                    [sin(2*pi/180) , cos(2*pi/180)]])
current_label = FLAGS.DefaultLabel
#===============

# Graphical
drawing = False # true if mouse is pressed
cropped = False
ix,iy = -1,-1
rx,ry = -1, -1
bboxes_thickness = 2
bboxes_text_size = 1
bboxes_text_width = 2
# MISC
img_index = 0

def rectangle_color(bbox_class):
    bbox_class = int(bbox_class)
    if bbox_class == 0:
        color = (255, 0, 0)

    elif bbox_class == 1:
        color = (0, 255, 0)

    elif bbox_class == 2:
        color = (0, 0, 255)

    elif bbox_class == 3:
        color = (128, 0, 0)

    elif bbox_class == 4:
        color = (128, 0, 128)

    elif bbox_class == 5:
        color = (32, 154, 0)

    elif bbox_class == 6:
        color = (0, 112, 112)

    elif bbox_class == 7:
        color = (60, 200, 200)

    elif bbox_class == 8:
        color = (10, 160, 255)

    elif bbox_class == 9:
        color = (255, 100, 0)

    else:
        color = (128, 128, 128)

    return color

# Mouse callback function
def draw(event,x,y,flags,param):
    global ix, iy, rx, ry, ldx, ldy, rux, ruy, drawing, img, DEFAULT, cropped , pathIMG

    cropped = False
    
    img = DEFAULT.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if bboxes != None:
                for i in range(len(bboxes)):
                    # cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][8]),bboxes_thickness)
                    cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), (255, 0, 0),bboxes_thickness)
            cv2.rectangle(img,(ix,iy),(x,y), rectangle_color(current_label), bboxes_thickness)
            # cv2.putText(img, str(current_label), (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, \
                # rectangle_color(current_label), bboxes_text_width, cv2.LINE_AA)
            rx, ry = x, y
            cv2.imshow(pathIMG, img)

        else:
            if bboxes != None:
                for i in range(len(bboxes)):
                    cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][4],bboxes[i][5]),(0,255,0),bboxes_thickness)
                    cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][6],bboxes[i][7]),(0,255,0),bboxes_thickness)
                    cv2.line(img,(bboxes[i][2],bboxes[i][3]),(bboxes[i][4],bboxes[i][5]),(0,255,0),bboxes_thickness)
                    cv2.line(img,(bboxes[i][2],bboxes[i][3]),(bboxes[i][6],bboxes[i][7]),(0,255,0),bboxes_thickness)
                    # cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]), rectangle_color(bboxes[i][8]), bboxes_thickness)
                    # cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][8]), bboxes_text_width, cv2.LINE_AA)
            # cv2.putText(img, str(current_label), (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(current_label), bboxes_text_width, cv2.LINE_AA)

            cv2.line(img,(0,y),(img.shape[1], y),(0,0,255),1)
            cv2.line(img,(x,0),(x, img.shape[0]),(0,255,255),1)
            cv2.imshow(pathIMG, img)

        # pass

    elif event == cv2.EVENT_LBUTTONUP:
        rx, ry = x, y 
        if ix > rx :
            ix, rx = rx, ix
        if iy > ry :
            iy, ry = ry , iy
        width = abs(ix - rx)
        height= abs(iy - ry)
        print(height)
        ldx = ix 
        ldy = iy + height
        rux = rx
        ruy = ry - height
        
        bboxes.append([ix, iy, rx, ry, ldx, ldy, rux, ruy])
        drawing = False
        
        print("bboxes ", bboxes)
        for i in range(len(bboxes)):
            cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),(0,255,0),bboxes_thickness)
            # cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][8]), bboxes_text_width, cv2.LINE_AA)

        cv2.imshow(pathIMG, img)



# Crop Image
def crop(ix,iy,x,y):
    global img, DEFAULT, cropped

    img = DEFAULT.copy()
    cv2.imshow(pathIMG,img)

    if (abs(ix - x) < abs(iy -y)):
        img = img[iy:y, ix:x]
    else:
        img = img[iy:y, ix:x]


def WriteboundingRect():
    global ix, iy, rx, ry,ldx, ldy, rux, ruy, pathIMG, bboxes,IMG
   
    line = ""
    
    for i in range(len(bboxes)):
        line = line + str(bboxes[i][0]) + ' ' + str(bboxes[i][1]) + '\n'+ str(bboxes[i][6]) + \
            ' ' + str(bboxes[i][7]) + '\n' + str(bboxes[i][2]) + ' ' +  str(bboxes[i][3]) + \
                '\n' + str(bboxes[i][4]) + ' ' +  str(bboxes[i][5]) + '\n'
    # line = str(Generic_location + "/" + pathIMG) + line  + '\n'
    #line =  line  + '\n'

    if not os.path.exists(pathIMG):
        os.makedirs(pathIMG)

    with open(pathSAV +IMG + 'cpos' + '.txt', 'a') as f:
        f.writelines(line)

    # with open(pathSAV + 'train' + "_" + current_time + '.txt', 'a') as f:
    #     f.writelines(line)
# Main Loop
def loop():
    global img, pathIMG, current_label,arcsin_len,arccos_lenx,delta_y,p
    
     
    cv2.namedWindow(pathIMG, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(pathIMG, draw)

    while (1):
        cv2.imshow(pathIMG, img)
        k = cv2.waitKey(1) & 0xFF

        if (k == 27):
            bboxes.clear()
            print("Cancelled Crop")
            break

        elif (k == ord('c')):
            img = DEFAULT.copy()

            if len(bboxes) != 0:
                bboxes.pop()

            for i in range(len(bboxes)):
                cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][4],bboxes[i][5]),(0,255,0),bboxes_thickness)
                cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][6],bboxes[i][7]),(0,255,0),bboxes_thickness)
                cv2.line(img,(bboxes[i][2],bboxes[i][3]),(bboxes[i][4],bboxes[i][5]),(0,255,0),bboxes_thickness)
                cv2.line(img,(bboxes[i][2],bboxes[i][3]),(bboxes[i][6],bboxes[i][7]),(0,255,0),bboxes_thickness)
                # cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),rectangle_color(bboxes[i][8]),bboxes_thickness)
                # cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][8]), bboxes_text_width, cv2.LINE_AA)

            cv2.imshow(pathIMG, img)

            print("bboxes ", bboxes)

            pass
        elif (k == ord('o')):
            p = p+1
        elif (k == ord('r')):
            img = DEFAULT.copy()
            
            print(p)

            # for i in range(len(bboxes)):
                
            rotation_cen = np.array([[(bboxes[p][0] + bboxes[p][2]) /2], [(bboxes[p][1] + bboxes[p][3]) /2]])
            print(rotation_cen)
            # width = abs(bboxes[0][0] - bboxes[0][2])
            # height= abs(bboxes[0][1] - bboxes[0][3])
            # diagonal_len = sqrt(width**2 + height**2)
            #print(rotation_cen)
            rotation_LU = np.array([[(bboxes[p][0]-rotation_cen[0][0])], [(bboxes[p][1]-rotation_cen[1][0])]])
            rotation_LD = np.array([[(bboxes[p][4]-rotation_cen[0][0])], [(bboxes[p][5]-rotation_cen[1][0])]])
            rotation_RD = np.array([[(bboxes[p][2]-rotation_cen[0][0])], [(bboxes[p][3]-rotation_cen[1][0])]])
            rotation_RU = np.array([[(bboxes[p][6]-rotation_cen[0][0])], [(bboxes[p][7]-rotation_cen[1][0])]])
            rotation_LU = np.dot(rotate_5, rotation_LU)
            rotation_LD = np.dot(rotate_5, rotation_LD)
            rotation_RD = np.dot(rotate_5, rotation_RD)
            rotation_RU = np.dot(rotate_5, rotation_RU)
            bboxes[p][0] = int(rotation_LU[0][0]+rotation_cen[0][0])
            bboxes[p][1] = int(rotation_LU[1][0]+rotation_cen[1][0])
            bboxes[p][2] = int(rotation_RD[0][0]+rotation_cen[0][0])
            bboxes[p][3] = int(rotation_RD[1][0]+rotation_cen[1][0])
            bboxes[p][4] = int(rotation_LD[0][0]+rotation_cen[0][0])
            bboxes[p][5] = int(rotation_LD[1][0]+rotation_cen[1][0])
            bboxes[p][6] = int(rotation_RU[0][0]+rotation_cen[0][0])
            bboxes[p][7] = int(rotation_RU[1][0]+rotation_cen[1][0])
            #print(bboxes)


            for i in range(len(bboxes)):
                # cv2.rectangle(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),rectangle_color(bboxes[i][4]),bboxes_thickness)
                cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][4],bboxes[i][5]),(0,255,0),bboxes_thickness)
                cv2.line(img,(bboxes[i][0],bboxes[i][1]),(bboxes[i][6],bboxes[i][7]),(0,255,0),bboxes_thickness)
                cv2.line(img,(bboxes[i][2],bboxes[i][3]),(bboxes[i][4],bboxes[i][5]),(0,255,0),bboxes_thickness)
                cv2.line(img,(bboxes[i][2],bboxes[i][3]),(bboxes[i][6],bboxes[i][7]),(0,255,0),bboxes_thickness)
                # cv2.putText(img, str(bboxes[i][4]), (bboxes[i][2] - 25, bboxes[i][3] - 10), cv2.FONT_HERSHEY_SIMPLEX, bboxes_text_size, rectangle_color(bboxes[i][8]), bboxes_text_width, cv2.LINE_AA)

            cv2.imshow(pathIMG, img)

            print("bboxes ", bboxes)

            pass

        elif (k == ord('s')):# and cropped):
            WriteboundingRect()
            print("======================================\n")
            print("                 ||                   ")
            print("                 ||                   ")
            print("                \||/                   ")
            print("                 \/                   \n")
            print("================Saved=================")
            print("Data : ", pathIMG)
            print("Label : ", bboxes)
            print("Number of Label :", len(bboxes))
            print("======================================\n")

            bboxes.clear()
            p = 0
            #save(img)
            break

    # print("Done!")
    cv2.destroyAllWindows()

# Iterate through images in path
def getIMG(path):
    global img, DEFAULT, pathIMG, IMG
    directory = os.fsencode(path)
    for filename in os.listdir(directory):
        # Get Image Path
        pathIMG = path + filename.decode("utf-8")
        print(pathIMG)
        # print(filename)
        IMG = filename.decode("utf-8").split('r',1)
        print(IMG)
        IMG = IMG[0]
        print(IMG)
        print("======================================")
        print("Current Data:", pathIMG)

        # Read Image
        img = cv2.imread(pathIMG,-1)
        
        DEFAULT = img.copy()

        # Draw image
        loop()

    return 0

# Main Function
def main():
    global img, DEFAULT

    if (pathDIR != ''):
        # Print Path
        print("pathDIR: " + pathDIR)

        # Cycle through files
        getIMG(pathDIR)

    elif (pathIMG != ''):
        # Print Path
        print("img: " + pathIMG)

        # Load Image
        img = cv2.imread(pathIMG,-1)
        DEFAULT = img.copy()

        # Draw Image
        loop()

# Run Main
if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
