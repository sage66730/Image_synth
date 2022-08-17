# -*- coding: utf-8 -*-
import csv
import cv2
import numpy as np
import time

#colors = [(0,0,255), (0,128,255), (0,255,255), (0,255,128), (0,204,0)]
colors = [(0,204,0), (0,255,128), (0,255,255), (0,128,255), (0,0,255)]
bs_names = ["EyeBlinkLeft",	"EyeLookDownLeft",	"EyeLookInLeft",	"EyeLookOutLeft",	"EyeLookUpLeft", "EyeSquintLeft",	"EyeWideLeft",	"EyeBlinkRight",	
            "EyeLookDownRight",	"EyeLookInRight",	"EyeLookOutRight",	"EyeLookUpRight",	"EyeSquintRight",	"EyeWideRight",	"JawForward",	"JawRight",	
            "JawLeft",	"JawOpen",	"MouthClose",	"MouthFunnel",	"MouthPucker",	"MouthRight",	"MouthLeft",	"MouthSmileLeft",	
            "MouthSmileRight",	"MouthFrownLeft",	"MouthFrownRight",	"MouthDimpleLeft",	"MouthDimpleRight",	"MouthStretchLeft",	"MouthStretchRight", "MouthRollLower",	
            "MouthRollUpper",	"MouthShrugLower",	"MouthShrugUpper",	"MouthPressLeft",	"MouthPressRight",	"MouthLowerDownLeft",	"MouthLowerDownRight",	"MouthUpperUpLeft",	
            "MouthUpperUpRight",	"BrowDownLeft",	"BrowDownRight",	"BrowInnerUp",	"BrowOuterUpLeft",	"BrowOuterUpRight",	"CheekPuff",	"CheekSquintLeft",	
            "CheekSquintRight",	"NoseSneerLeft",	"NoseSneerRight",	"TongueOut",	"HeadYaw",	"HeadPitch",	"HeadRoll",	"LeftEyeYaw",	
            "LeftEyePitch",	"LeftEyeRoll",	"RightEyeYaw",	"RightEyePitch",	"RightEyeRoll"]

def MSE(pred, label):
    return sum((pred - label) ** 2)

def WeightedMSE_M(pred, label, alpha):
    return alpha*(sum((pred[18:41] - label[18:41]) ** 2)) + (1-alpha)*(sum((pred[:18] - label[:18]) ** 2)+sum((pred[41:61] - label[41:61]) ** 2))

def load_csv(csv_path):
    fp = open(csv_path, newline='')
    cdata = csv.reader(fp)

    temp = []
    for idx, row in enumerate(cdata):
        if idx != 0:
            bs = [float(num) for num in row[2:]]
            temp.append(bs)
    temp = np.array(temp)
    return temp  

def interpolation(l1, l2):
    if l1 == l2: return [idx for idx in range(l2)]
    assert l1 > l2
    m = l1-1
    n = l2-1
    arr = list(range(l2))
    arr = [int( (idx*m)/n ) for idx in arr]
    assert len(arr) == len(set(arr))
    return arr 

def getColors(base_csv, model_csv):
    code = []
    for i in range(len(base_csv)):
        temp = []
        for j in range(61):
            diff = abs(base_csv[i][j]-model_csv[i][j])
            if diff <= 0.01: temp.append(0)
            elif diff <= 0.05: temp.append(1)
            elif diff <= 0.1: temp.append(2)
            elif diff <= 0.5: temp.append(3)
            else: temp.append(4)
        code.append(temp)

    return code 

def combine(f1, f2):
    buffer = np.concatenate((f1, f2), axis=2)
    return buffer

def readFrames(cap):
    frames = []
    status, img = cap.read()
    while status:
        frames.append(cv2.resize(img, None, fx=0.5, fy=0.5))
        status, img = cap.read()
    frames = np.array(frames)
    return frames

def putBlendShape(img, bs, x, y, color_fn):
    for i in range(30):
         cv2.putText(img, f"{bs_names[i]}:{bs[i]}", (x, y+15*(i+1)), cv2.FONT_ITALIC, 0.4, color_fn(i), 1)
    for i in range(30, 61):
         cv2.putText(img, f"{bs_names[i]}:{bs[i]}", (x+650, y+15*(i-30)), cv2.FONT_ITALIC, 0.4, color_fn(i), 1)
    return 

def show(idx, frames, base_bs, model_bs, loss, color_code):
    img = frames[idx]
    cv2.putText(img, f"frame {idx}", (img.shape[1]//2-100, 30), cv2.FONT_ITALIC, 1, (0, 0, 0), 2)
    cv2.putText(img, f"up:speed_up,  down:speed_down,  left:prev_frame,  right:next_frame,  enter:pause/continue,  esc:quit", (10, img.shape[0]-10), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1) 
    cv2.putText(img, "MSE:{:.3f} wMSE_M08:{:.3f}".format(loss["MSE"][idx], loss["wMSE_M08"][idx]), (img.shape[1]//2+10, img.shape[0]-10), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)    
    
    cv2.putText(img, f"Baseline:", (10, 50), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1)
    putBlendShape(img, base_bs[idx], 10, 50, lambda x: (0,0,0))  
    cv2.putText(img, f"Model:", (img.shape[1]//2+10, 50), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1) 
    putBlendShape(img, model_bs[idx], img.shape[1]//2+10, 50, lambda x: colors[color_code[idx][x]])  
    
    cv2.imshow("comparison", img)
    return 

def main():
    # load two livelink csv files 
    base_csv = load_csv("base.csv")
    model_csv = load_csv("Ex_wMSE.csv")

    # load two videos
    base_cap = cv2.VideoCapture('base.mkv')
    base_frames = readFrames(base_cap)
    model_cap = cv2.VideoCapture('Ex_wMSE.mkv')
    model_frames = readFrames(model_cap)
    print(f"base img size: {base_frames.shape}\nmodel img size: {model_frames.shape}") 

    # align number of frames
    print(f"number of frames:\nbase_video:{base_frames.shape[0]}\nbase_csv:{base_csv.shape[0]}\nmodel_video:{model_frames.shape[0]}\nmodel_csv:{model_csv.shape[0]}")
    min_length = min(base_frames.shape[0], base_csv.shape[0], model_frames.shape[0], model_csv.shape[0])

    idx = interpolation(base_csv.shape[0], min_length)
    base_csv = base_csv[idx]
    idx = interpolation(model_csv.shape[0], min_length)
    model_csv = model_csv[idx]

    idx = interpolation(base_frames.shape[0], min_length)
    base_frames = base_frames[idx]
    idx = interpolation(model_frames.shape[0], min_length)
    model_frames = model_frames[idx]

    print(f"number of frames (aligned):\nbase_video:{base_frames.shape[0]}\nbase_csv:{base_csv.shape[0]}\nmodel_video:{model_frames.shape[0]}\nmodel_csv:{model_csv.shape[0]}")

    # calculate loss
    loss = dict()
    loss["MSE"] = [MSE(base_csv[i], model_csv[i]) for i in range(min_length)]
    loss["wMSE_M08"] = [WeightedMSE_M(base_csv[i], model_csv[i], 0.8) for i in range(min_length)]

    # calculate color
    color_code = getColors(base_csv, model_csv)

    # combine two videos side by side
    frames = combine(base_frames, model_frames)

    # show
    idx = 0
    lap_time=0.1   
    while True:
        show(idx, frames, base_csv, model_csv, loss, color_code)

        # normal display mode
        nextKey = cv2.waitKeyEx(1) 
        if nextKey == 27:
            # esc
            break
        elif nextKey == 2490368: 
            # up arrow
            lap_time /= 2
        elif nextKey == 2621440: 
            # down arrow
            lap_time *= 2
 
        # under pause mode (press enter)
        elif nextKey == 13:    
            while True:
                nextKey=cv2.waitKeyEx(0) 
                if nextKey == 13: 
                    break 
                elif nextKey == 27:
                    base_cap.release()
                    model_cap.release()
                    cv2.destroyAllWindows()
                    return 
                elif nextKey == 2424832: 
                    # left arrow
                    idx = (idx-1) % len(frames)
                    show(idx, frames, base_csv, model_csv, loss, color_code)
                elif nextKey == 2555904: 
                    # right arrow
                    idx = (idx+1) % len(frames)
                    show(idx, frames, base_csv, model_csv, loss, color_code)

        time.sleep(lap_time)  
        idx = (idx+1) % len(frames)

    base_cap.release()
    model_cap.release()
    cv2.destroyAllWindows()
    return 

if __name__ == "__main__":
    main()
 
