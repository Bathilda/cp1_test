import cv2
import numpy as np
import time

class Detection():
    def __init__(self, weights_path: str, config_path: str):
        self.weights_path = weights_path
        self.config_path = config_path
        self.cv_net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.labels = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' }
        
    def get_detected_img(self, img_array, conf_threshold = 0.5, nms_threshold = 0.4, is_print = True):
        
        # 원본 이미지를 네트웍에 입력시에는 (416, 416)로 resize 함. 
        # 이후 결과가 출력되면 resize된 이미지 기반으로 bounding box 위치가 예측 되므로 이를 다시 원복하기 위해 원본 이미지 shape정보 필요
        rows = img_array.shape[0]
        cols = img_array.shape[1]
        
        draw_img = img_array.copy()
        
        #전체 Darknet layer에서 13x13 grid, 26x26, 52x52 grid에서 detect된 Output layer만 filtering
        layer_names = self.cv_net.getLayerNames()
        # opencv DNN이 업그레이드 되면서 cv_net_yolo.getUnconnectedOutLayers()의 반환 결과가 2차원이 아니라 1차원 형태로 layer id가 반환됩니다. 
        # 따라서 아래 코드는 강의 영상의 layers_names[i[0] - 1]에서 layers_names[i - 1] 로 변경합니다. 2022.08.09
        outlayer_names = [layer_names[i - 1] for i in self.cv_net.getUnconnectedOutLayers()]
        
        # 로딩한 모델은 Yolov3 416 x 416 모델임. 원본 이미지 배열을 사이즈 (416, 416)으로, BGR을 RGB로 변환하여 배열 입력
        self.cv_net.setInput(cv2.dnn.blobFromImage(img_array, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False))
        start = time.time()
        # Object Detection 수행하여 결과를 cvOut으로 반환 
        cv_outs = self.cv_net.forward(outlayer_names)
        layerOutputs = self.cv_net.forward(outlayer_names)
        # bounding box의 테두리와 caption 글자색 지정
        green_color=(0, 255, 0)
        red_color=(0, 0, 255)

        class_ids = []
        confidences = []
        boxes = []

        # 3개의 개별 output layer별로 Detect된 Object들에 대해서 Detection 정보 추출 및 시각화 
        for ix, output in enumerate(cv_outs):
            # Detected된 Object별 iteration
            for jx, detection in enumerate(output):
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # confidence가 지정된 conf_threshold보다 작은 값은 제외 
                if confidence > conf_threshold:
                    #print('ix:', ix, 'jx:', jx, 'class_id', class_id, 'confidence:', confidence)
                    # detection은 scale된 좌상단, 우하단 좌표를 반환하는 것이 아니라, detection object의 중심좌표와 너비/높이를 반환
                    # 원본 이미지에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산
                    center_x = int(detection[0] * cols)
                    center_y = int(detection[1] * rows)
                    width = int(detection[2] * cols)
                    height = int(detection[3] * rows)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    # 3개의 개별 output layer별로 Detect된 Object들에 대한 class id, confidence, 좌표정보를 모두 수집
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        # NMS로 최종 filtering된 idxs를 이용하여 boxes, classes, confidences에서 해당하는 Object정보를 추출하고 시각화.
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(idxs) > 0:
            for i in idxs.flatten():
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                # labels_to_names 딕셔너리로 class_id값을 클래스명으로 변경. opencv에서는 class_id + 1로 매핑해야함.
                caption = "{}: {:.4f}".format(self.labels[class_ids[i]], confidences[i])
                #cv2.rectangle()은 인자로 들어온 draw_img에 사각형을 그림. 위치 인자는 반드시 정수형.
                cv2.rectangle(draw_img, (int(left), int(top)), (int(left+width), int(top+height)), color=green_color, thickness=2)
                cv2.putText(draw_img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)

        if is_print:
            print('Detection 수행시간:',round(time.time() - start, 2),"초")
        return draw_img
    
    def get_video(self, video_path, output_path, conf_threshold = 0.5, nms_threshold = 0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        vfile = cv2.VideoCapture(video_path)

        codec =cv2.VideoWriter_fourcc(*'XVID')

        vid_size = (round(vfile.get(cv2.CAP_PROP_FRAME_WIDTH)),round(vfile.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        vid_fps = vfile.get(cv2.CAP_PROP_FPS)

        vid_writer = cv2.VideoWriter(output_path, codec, vid_fps, vid_size) 
        
        frame_cnt = int(vfile.get(cv2.CAP_PROP_FRAME_COUNT))
        print('총 Frame 갯수:', frame_cnt)

        while True:
            vret, img = vfile.read()
            if not vret:
                print('더 이상 처리할 frame이 없습니다.')
                break
            img = self.get_detected_img(img, conf_threshold=conf_threshold, nms_threshold=nms_threshold, is_print=True)
            vid_writer.write(img)
        
        vid_writer.release()
        vfile.release()