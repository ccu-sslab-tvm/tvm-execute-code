class yolov5_704_int8:
    #image
    img = [
        '202205041634588.jpeg',
        '202205051553300.jpeg',
        '202205051553304.jpeg',
        '202205051553306.jpeg',
        '202205051553337.jpeg',
        'sock2_gray_192x192.jpg'
    ]

    # model info
    name = 'yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite'
    img_height_width = 192
    input_name = 'input_1:int8'
    input_shape = (1, img_height_width, img_height_width, 1)
    input_dtype = 'int8'

    # dequantance info
    dequantance = [0.006245302967727184, 122]
    candidate = 540
    class_num = 5
    class_label = ['1','2','3','4','5']

    def __init__(self):
        self.input_name = self.input_name.replace(':', '_')

class y7_1336_int8:
    #image
    img = [
        '1676530722154.jpg',
        'hand2.bmp'
    ]

    # model info
    name = 'y7_1336_p-0.9349_r-0.9556_map50-0.8342_160x160_ch1-int8.tflite'
    img_height_width = 160
    input_name = 'serving_default_input_1:0_int8'
    input_shape = (1, img_height_width, img_height_width, 1)
    input_dtype = 'int8'

    # dequantance info
    dequantance = [0.0354100801050663, 128]
    candidate = 375
    class_num = 6
    class_label = ['single&palm', 'single&back', 'open', 'fist', 'folded', 'fist_back']

    def __init__(self):
        self.input_name = self.input_name.replace(':', '_')
        return