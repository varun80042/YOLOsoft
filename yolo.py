import cv2
import numpy as np
import json
import sys

if len(sys.argv) != 2:
    print("Run using - yolo.exe sample.mp4")
    sys.exit(1)

video_file = sys.argv[1]

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
confidence_threshold = 0.5
nms_threshold = 0.2

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(video_file)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

detected_objects = []

frame_number = 0

while True:
    success, img = cap.read()

    if not success:
        break  

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    class_ids = []
    confidences = []
    bbox = []

    for output in layerOutputs:
        for detections in output:
            scores = detections[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detections[0] * width)
                center_y = int(detections[1] * height)
                w = int(detections[2] * width)
                h = int(detections[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bbox.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(bbox, confidences, confidence_threshold, nms_threshold)

    for i in indices.flatten():
        x, y, w, h = bbox[i]
        label = str(classes[class_ids[i]])
        confidence = round(confidences[i], 2)
        timestamp = frame_number / fps  
        detected_objects.append({"label": label, "confidence": confidence, "coordinates": {"x": x, "y": y, "width": w, "height": h}, "timestamp": timestamp})
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(img, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(img) 
    frame_number += 1  

cap.release()
out.release()
cv2.destroyAllWindows()

with open("output.json", "w") as json_file:
    json.dump(detected_objects, json_file, indent=4)

