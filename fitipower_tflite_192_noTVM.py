'''
使用模型: yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite

這個檔案沒有經過 TVM
用於測試天鈺模型之輸出結果
By 陳洧成

目前已知問題： 無
'''

import os
from tkinter import W, Y
import cv2
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

np.set_printoptions(threshold=sys.maxsize)

# class_index = [
#                "socks" : 0
#                "powerCable" : 1
#                "petstool" : 2
#                "slipper" : 3
#                "Scale" : 4
#                ]


img_name = '202205051553300.jpeg'
test_image = './img/' + img_name
output_path = './test_outputs/fitipower_tflite_192_noTVM/'

output_info = False

model_path = './model/yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite'

if not os.path.exists(output_path):
    os.mkdir(output_path)

output_path = output_path + img_name

if not os.path.exists(output_path):
    os.mkdir(output_path)

def model_load(model_path,ori_img):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()
    input_scale,input_zero_point = input_detail[0]['quantization']
    output_detail = interpreter.get_output_details()

    test_img = (ori_img+input_zero_point)
    print("test_img is int8: ",np.issubdtype(test_img.dtype,np.int8))

    img_expand = np.expand_dims(test_img,axis=0)
    img_expand = np.expand_dims(img_expand,axis=-1)
    if output_info:
        f = open(output_path + "/expand.txt",'w')
        f.write(str(img_expand))
        f.close()
    print("input shape: ",img_expand.shape)
    img_expand = img_expand.astype("int8")

    print("img_expand is int8: ",np.issubdtype(img_expand.dtype,np.int8))

    interpreter.set_tensor(input_detail[0]['index'],img_expand)

    time_start = datetime.now()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_detail[0]['index'])
    time_end = datetime.now() # 計算 graph_mod 的執行時間
    print("spent {0}", time_end - time_start)

    print("output_data is int8: ",np.issubdtype(output_data.dtype,np.int8))

    # output_data = output_data.astype("float")
    if output_info:
        f = open(output_path + "/out.txt",'w')
        f.write(str(output_data))
        f.close()
    print(output_data.shape)
    output_data = np.squeeze(output_data)
    print(output_data.shape)
    return output_data

def DEQNT(a):
    return 0.006245302967727184 * (a + 122)

def clamp(val,min,max):
    if val > min:
        if val<max:
            return val
        else: 
            return max
    else:
        return min

def iou(box1,box2):
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    x11 = max(box1[0],box2[0])
    y11 = max(box1[1],box2[1])
    x22 = max(box1[2],box2[2])
    y22 = max(box1[3],box2[3])

    intersection = (x22-x11+1)*(y22-y11+1)
    return intersection/(area2+area1-intersection)

def nms(vecBox,threshold):
    vecbox_sorted = sorted(vecBox,key=lambda vecBox : vecBox[4])
    picked_bbox = []
    while(len(vecbox_sorted) > 0):
        picked_bbox.append(vecbox_sorted[-1])
        vecbox_sorted.pop()
        i = 0
        while (i < len(vecbox_sorted)):
            iou_score = iou(picked_bbox[-1],vecbox_sorted[i])
            if (iou_score >= threshold):
                vecbox_sorted.pop(i)
                continue
            i = i+1

    return picked_bbox

def box_find(output_data):
    num_candidate_box = 540
    num_class = 5
    dimensions = 10
    confidence_threshold = 0.3
    score_threshold = 0.4
    w_and_h = 192

    data = output_data

    box = [] # x1, y1, x2, y2, maxclassscore, maxClassId

    for i in range(num_candidate_box):  # num_candidate_box = 540
        confidence = DEQNT(data[i][4]) # 把candidate_box的confidence取出來
        if (confidence >= confidence_threshold): # 用剛剛那個confidence去跟confidence_threshold比較，如果大於這個threshold，就去比較每一個class的score，記下score大的那個class
            maxclassid = -1
            maxclassscore = 0
            for j in range(num_class):
                score = DEQNT(data[i][j+num_class])
                if(score > maxclassscore):
                    maxclassid = j
                    maxclassscore = score
            if (maxclassscore > score_threshold): # 如果剛剛留下的那個class的score有大於score_threshold的話，就把這個candidate_box的xywh讀出來
                x = DEQNT(data[i][0])
                y = DEQNT(data[i][1])
                w = DEQNT(data[i][2])
                h = DEQNT(data[i][3])
                x1 = (int)(clamp((x-w/2)*w_and_h,0,w_and_h))
                y1 = (int)(clamp((y-h/2)*w_and_h,0,w_and_h))
                x2 = (int)(clamp((x+w/2)*w_and_h,0,w_and_h))
                y2 = (int)(clamp((y+h/2)*w_and_h,0,w_and_h))
                box.append([x1, y1, x2, y2, maxclassscore, maxclassid])
    return box

def box_select(box):
    iou_threshold = 0.3
    num_class = 5
    w_and_h = 192
    isobject = 0
    max_class_id = 0
    max_confidence = 0
    dist_y = 0

    show = []

    for i in range(num_class):
        class_box = []
        for j in range(len(box)):
            if (box[j][5]==i):
                class_box.append(box[j])
        if (len(class_box) > 0):
            result = nms(class_box,iou_threshold)
            if (len(result)>0):  # result [x1, y1, x2, y2, maxclassscore, maxclassid]
                for j in range(len(result)):
                    show.append([result[j][0], result[j][1], result[j][2], result[j][3], result[j][4], result[j][5]])
                    isobject = 1
    return show


if __name__ == "__main__":
    ori_img = cv2.imread(test_image,cv2.IMREAD_GRAYSCALE)
    ori_img = cv2.resize(ori_img, (192, 192))
    if output_info:
        f = open(output_path + "/in.txt",'w')
        f.write(str(ori_img))
        f.close()

    # model_load會把模型load近來，然後把圖片餵進去，回傳運算之後的結果
    output_data = model_load(model_path,ori_img)

    # box_find會把剛剛的運算結果透過confidence與各個class的score來篩出一些bbox
    box = box_find(output_data)
    if output_info:
        f = open(output_path + "/box.txt",'w')
        f.write(str(box))
        f.close()

    # 把最後的bbox顯示出來
    show = box_select(box)

    label = ["socks", "powercable", "petstool", "slipper", "scale"]
    if (len(show)>0):
        for i in range(len(show)):
            print("class:", show[i][5], ", x1:",show[i][0], ", y1:",show[i][1], ", x2:",show[i][2], ", y2:",show[i][3], ", confidence:",show[i][4]*100)
            ori_img = cv2.rectangle(ori_img,(show[i][0],show[i][1]),(show[i][2],show[i][3]),(36,255,12),2)
            label_size = cv2.getTextSize(label[show[i][5]], cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            cv2.rectangle(ori_img, (show[i][0], show[i][1]), (show[i][0]+label_size[0][0], show[i][1]-label_size[0][1]), (255, 255, 255), cv2.FILLED)
            cv2.putText(ori_img, label[show[i][5]], (show[i][0], show[i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5 , (0, 0, 0), 1)
    else:
        print("no object")
    
    cv2.imwrite(output_path + '/image.jpg', ori_img)

    print("show: ",show)