import cv2

from post_process import post_process_fitipower


class tflite_5x5_cus_model_fp32:
    # input data
    input_num = [
        [
            [62, 49, 2, 51, 89],
            [15, 8, 13, 11, 40],
            [93, 94, 31, 36, 26],
            [70, 35, 80, 66, 44],
            [20, 95, 77, 49, 76]
        ],
    ]

    # model info
    framework = 'tflite'
    name = '_5x5_tfmodel_fp32.tflite'
    input_name = 'serving_default_input_1:0'
    input_shape = (1, 5, 5, 1)
    input_dtype = 'float32'
    flags = None

    # dequantance info
    dequantance = [1, 0]

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
        self.input_name = self.input_name.replace(':', '_')

class tflite_5x5_cus_model_int8:
    # input data
    input_num = [
        [
            [62, 49, 2, 51, 89],
            [15, 8, 13, 11, 40],
            [93, 94, 31, 36, 26],
            [70, 35, 80, 66, 44],
            [20, 95, 77, 49, 76]
        ],
    ]

    # model info
    framework = 'tflite'
    name = '_5x5_tfmodel_int8.tflite'
    input_name = 'serving_default_input_1:0'
    input_shape = (1, 5, 5, 1)
    input_dtype = 'int8'
    flags = None

    # dequantance info
    dequantance = [4.9490203857421875, 128]

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """

class tflite_sine_float32:
    # input data
    input_num = [
        0.5,
    ]

    # model info
    framework = 'tflite'
    name = 'sine_model.tflite'
    input_name = 'dense_4_input'
    input_shape = (1, 1)
    input_dtype = 'float32'
    flags = None

    # dequantance info
    dequantance = [1, 0]

class tflite_yolov5_704_int8:
    # input data
    img = [
        '202205041634588.jpeg',
        '202205051553300.jpeg',
        '202205051553304.jpeg',
        '202205051553306.jpeg',
        '202205051553337.jpeg',
        'sock2_gray_192x192.jpg'
    ]

    # model info
    framework = 'tflite'
    name = 'yolov5_704_p-0.9474_r-0.9408_map50-0.6404_192x192_ch1_ReLU-int8.tflite'
    img_height_width = 192
    input_name = 'input_1:int8'
    input_shape = (1, img_height_width, img_height_width, 1)
    input_dtype = 'int8'
    flags = cv2.IMREAD_GRAYSCALE

    # dequantance info
    dequantance = [0.006245302967727184, 122]
    candidate = 540
    class_num = 5
    class_label = ['1','2','3','4','5']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
        self.input_name = self.input_name.replace(':', '_')
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )

class tflite_y7_1336_int8:
    # input data
    img = [
        '1676530722154.jpg',
        'hand2.bmp'
    ]

    # model info
    framework = 'tflite'
    name = 'y7_1336_p-0.9349_r-0.9556_map50-0.8342_160x160_ch1-int8.tflite'
    img_height_width = 160
    input_name = 'serving_default_input_1:0_int8'
    input_shape = (1, img_height_width, img_height_width, 1)
    input_dtype = 'int8'
    flags = cv2.IMREAD_GRAYSCALE

    # dequantance info
    dequantance = [0.0354100801050663, 128]
    candidate = 375
    class_num = 6
    class_label = ['single&palm', 'single&back', 'open', 'fist', 'folded', 'fist_back']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )

class tflite_yolov5n_fp32:
    # input data
    img = [
        'tourist_area.jpg',
        'bus.jpg',
        'zidane.jpg'
    ]

    # model info
    framework = 'tflite'
    name = 'yolov5n-fp16.tflite'
    img_height_width = 640
    input_name = 'serving_default_input_1:0'
    input_shape = (1, img_height_width, img_height_width, 3)
    input_dtype = 'float32'
    flags = None

    # dequantance info
    dequantance = [1, 0]
    candidate = 25200
    class_num = 80
    class_label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )

class torchscript_yolov5n_fp32:
    # input data
    img = [
        'tourist_area.jpg',
        'bus.jpg',
        'zidane.jpg'
    ]

    # model info
    framework = 'torchscript'
    name = 'yolov5n.torchscript'
    img_height_width = 640
    input_name = 'serving_default_input_1:0'
    input_shape = (1, 3, img_height_width, img_height_width)
    input_dtype = 'float32'
    flags = None

    # dequantance info
    dequantance = [1, 0]
    candidate = 25200
    class_num = 80
    class_label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )

class tflite_yolov5n_int8:
    # input data
    img = [
        'tourist_area.jpg',
        'bus.jpg',
        'zidane.jpg'
    ]

    # model info
    framework = 'tflite'
    name = 'yolov5n-int8.tflite'
    img_height_width = 640
    input_name = 'serving_default_input_1:0'
    input_shape = (1, img_height_width, img_height_width, 3)
    input_dtype = 'int8'
    flags = None

    # dequantance info
    dequantance = [0.004483491647988558 , -3]
    candidate = 25200
    class_num = 80
    class_label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )

class tflite_yolov5x_fp32:
    # input data
    img = [
        'tourist_area.jpg',
        'bus.jpg',
        'zidane.jpg'
    ]

    # model info
    framework = 'tflite'
    name = 'yolov5x-fp16.tflite'
    img_height_width = 640
    input_name = 'serving_default_input_1:0'
    input_shape = (1, img_height_width, img_height_width, 3)
    input_dtype = 'float32'
    flags = None

    # dequantance info
    dequantance = [1, 0]
    candidate = 25200
    class_num = 80
    class_label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )

class tflite_yolov5x_int8:
    # input data
    img = [
        'tourist_area.jpg',
        'bus.jpg',
        'zidane.jpg'
    ]

    # model info
    framework = 'tflite'
    name = 'yolov5x-int8.tflite'
    img_height_width = 640
    input_name = 'serving_default_input_1:0'
    input_shape = (1, img_height_width, img_height_width, 3)
    input_dtype = 'int8'
    flags = None

    # dequantance info
    dequantance = [0.0044934190809726715, -1]
    candidate = 25200
    class_num = 80
    class_label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        """ if input name have `:`, change it to `_` or not. """
    
    def post_process(self, output, output_path, img_path, img_selection):
        post_process_fitipower.post_process(
            output[0],
            output_path,
            img_path,
            img_selection,
            self.flags,
            self.img_height_width,
            self.dequantance,
            self.candidate,
            self.class_num,
            self.class_label
        )
