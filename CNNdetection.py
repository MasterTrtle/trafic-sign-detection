# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:24:33 2024

@author: Meiyiguang
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import selectivesearch
from detection import build_detection_validation
# dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = self.load_data()

    def load_data(self):
        images = []
        for img_name in os.listdir(self.data_dir):
            img_path = os.path.join(self.data_dir, img_name)
            label = self.extract_label_from_filename(img_name)
            label_idx = label_to_idx[label]  # 将标签转换为索引
            images.append((img_path, label_idx))
        return images

    def extract_label_from_filename(self, filename):
        parts = filename.split('_')
        label = parts[2]  # 
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label_idx = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label_idx


label_to_idx = {'danger': 0, 'interdiction': 1, 'ceder': 2, 'obligation': 3, 'frouge': 4, 'fvert': 5, 'forange': 6, 'stop': 7,'none': 8}
classes = ['danger', 'interdiction', 'ceder', 'obligation', 'frouge', 'fvert', 'forange', 'stop','none']
#CNN
class CNN(nn.Module):
    def __init__(self, num_classes=9):  
        super(CNN, self).__init__()
        ##vgg16 accuracy too low
        """
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
       """
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ResNet18 
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=9):
        super(CustomResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class Sign_Classifier():
    def __init__(self):
        self.net = CustomResNet18(num_classes=9)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001) #learning_rate = 0.001
    
    def fit(self, train_dataset,val_dataset, n_epoch=20, batch_size=32): #batch_size = 32 num_epochs = 10
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        total_step = len(train_loader)
        for epoch in range(n_epoch):
            self.net.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Initialise le gradient courant à zéro
                self.optimizer.zero_grad()
                # Prédit la classe des éléments X du batch courant
                labels_pred = self.net(images)
                # Calcul la fonction de coût sur ce batch
                loss = self.criterion(labels_pred, labels)
                # Rétropropagation pour calculer le gradient
                loss.backward()
                # Effectue un pas de la descente de gradient
                self.optimizer.step()

                if (batch_idx+1) % n_epoch == 0:
                    print(f'Training: Epoch [{epoch+1}/{n_epoch}], Step [{batch_idx+1}/{total_step}], Loss: {loss.item():.4f}')
             
            self.net.eval()  
            with torch.no_grad():
                 correct = 0
                 total = 0
                 for images, labels in val_loader:
                     labels_pred = self.net(images)
                     _, predicted = torch.max(labels_pred.data, 1)
                     total += labels.size(0)
                     correct += (predicted == labels).sum().item()

                 accuracy = 100 * correct / total
                 print(f'Validation: Epoch [{epoch+1}/{n_epoch}], Accuracy: {accuracy:.2f}%')    
        
        save_path = 'traffic_sign_cnn.pth'
        torch.save(self.net.state_dict(), save_path)

        print(f'Model saved at: {os.path.abspath(save_path)}')
        
        return self
    
    def predict(self,roi):
        self.net.eval()
        roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = transform(roi).unsqueeze(0) 
       
        with torch.no_grad():
            labels_pred = self.net(roi_tensor)
            probabilities = torch.softmax(labels_pred, dim=1).numpy()[0]
            np.set_printoptions(precision=2, suppress=True)
            #print(probabilities)
            confidence = probabilities[labels_pred.argmax().item()]
            _, predicted = torch.max(labels_pred.data, 1)
            predicted_label = classes[predicted.item()]
        return predicted_label,probabilities,confidence
      
      
    def load_model(self,model_path):
          self.net.load_state_dict(torch.load(model_path))
          self.net.eval()
          return self.net
          

#model_path = 'C:/Users/23158/traffic_sign_cnn.pth' 
#clf = Sign_Classifier()
#clf.load_model(model_path)


#%%train
"""
train_dataset = TrafficSignDataset(data_dir='C:/Users/23158/Desktop/SY32PROJET/dataset2/train/cropped_images', transform=transform)
val_dataset = TrafficSignDataset(data_dir='C:/Users/23158/Desktop/SY32PROJET/dataset2/val/cropped_images', transform=transform)

clf = Sign_Classifier()
clf.fit(train_dataset, val_dataset)

"""
    



#%%load
"""
model_path = 'traffic_sign_cnn.pth' 
clf = Sign_Classifier()
clf.load_model(model_path)
"""
#%%select_region_and_predict
def select_region_and_predict(image_path, Sign_Classifier):
    global ref_point, cropping, image, clone
    cropping = False
    ref_point = []

    def click_and_crop(event, x, y, flags, param):
        global ref_point, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cropping = False

            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

    image = cv2.imread(image_path)
    clone = image.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            image = clone.copy()

        elif key == ord('c') and len(ref_point) == 2:
            roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            prediction,_,confidence = Sign_Classifier.predict(roi)

            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            text = f"Prediction: {prediction}"
            cv2.putText(image, text, (ref_point[0][0], ref_point[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("image", image)
            ref_point = []

        elif key == ord('0'):
            break

    cv2.destroyAllWindows()

    
#image_path = 'C:/Users/23158/Desktop/SY32PROJET/dataset2/train/images/0833.jpg'  #
#select_region_and_predict(image_path,clf)
#%%selective_search
"""
def sliding_window(image, min_window_size, max_window_size, step):
    for window_size in range(max_window_size, min_window_size - 1, -step):
        for y in range(0, image.shape[0] - window_size + 1, step):
            for x in range(0, image.shape[1] - window_size + 1, step):
                yield (x, y, window_size, window_size, image[y:y + window_size, x:x + window_size])

def detect_traffic_signs(image_path, model, min_window_size=100, max_window_size=500, step=25, threshold=0.90):
    image = cv2.imread(image_path)
    clone = image.copy()
    detections = []

    for (x, y, w, h, window) in sliding_window(image, min_window_size, max_window_size, step):
            predicted_class,probabilities,confidence=model.predict(window)
            
            if predicted_class != 'none' and confidence > threshold:
                detections.append((x, y, w, h, predicted_class, confidence))
    print(detections)            

    filtered_detections = non_max_suppression(detections)

    for (x, y, w, h, predicted_class, confidence) in filtered_detections:
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{predicted_class} ({confidence:.2f})"
        cv2.putText(clone, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))  
    plt.title('Detections')  
    plt.axis('off')  
    plt.show()

def non_max_suppression(detections, overlap_thresh=0):
    if len(detections) == 0:
        return []

    x1 = np.array([det[0] for det in detections])
    y1 = np.array([det[1] for det in detections])
    x2 = np.array([det[0] + det[2] for det in detections])
    y2 = np.array([det[1] + det[3] for det in detections])
    confidences = np.array([det[5] for det in detections])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(confidences)[::-1]

    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[idxs[1:]]

        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    return [detections[i] for i in keep]

"""

###without piramid
def sliding_window_without_piramid(image, step_size_ratio, min_window_size_ratio, max_window_size_ratio):
    height, width = image.shape[:2]
    print("imagesize:",width,height)
    min_window_width = int(width * min_window_size_ratio)
    max_window_width = int(width * max_window_size_ratio)
    size_ratio=max_window_size_ratio-0.1
    #scale_factor = 0.9
    
    window_width = int( max_window_width*size_ratio)
    window_height = window_width
    
    
    while window_width >= 150:  
        print("windowsize:",window_width,window_height)
        #step_size = 50
        step_size = int(window_width * step_size_ratio)
        print("step_size:",step_size)
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, step_size):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        size_ratio=size_ratio-0.1
        window_width = int( max_window_width*size_ratio)
        window_height = window_width
        
def sliding_window_without_piramid_feu(image, step_size_ratio, min_window_size_ratio, max_window_size_ratio):
    height, width = image.shape[:2]
    print("imagesize:",width,height)
    min_window_height = int(height * min_window_size_ratio)
    max_window_height = int(height * max_window_size_ratio)
    size_ratio=max_window_size_ratio-0.1
    #scale_factor = 0.9
    
    window_height = int(max_window_height*size_ratio)
    window_width = int(window_height/2.2)
    
    
    while window_height >= 200:  
        print("windowsize:",window_width,window_height)
        #step_size = 50
        step_size = int(window_height * step_size_ratio)
        print("step_size:",step_size,int(step_size/2.2))
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, int(step_size/2)):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        size_ratio=size_ratio-0.1
        window_height =  int(max_window_height*size_ratio)
        window_width = int(window_height/2.2)

def detect_traffic_signs_without_piramid(image, model, step_size_ratio=0.2, min_window_size_ratio=0.1, max_window_size_ratio=1.0):
    detections = []
   
    for (x, y, (win_width, win_height), window) in sliding_window_without_piramid(image, step_size_ratio,min_window_size_ratio,max_window_size_ratio):
        if window.shape[0] != win_height or window.shape[1] != win_width:
            continue
        prediction,prediction_probabilities,confidence=model.predict(window)
        #prediction = model.predict_window(window)
        #prediction_probabilities = model.predict_proba_window(window)
        #max_probabilities = np.max(prediction_probabilities, axis=1)
        
        if prediction != 'none' and prediction != 'frouge' and prediction != 'forange' and prediction != 'fvert' and confidence >=0.99:  # Assuming '1' indicates a traffic sign
            x_orig = x
            y_orig = y
            w_orig = win_width
            h_orig = win_height
            prob = confidence
            detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, prediction, prob))
            
    for (x, y, (win_width, win_height), window) in sliding_window_without_piramid_feu(image, step_size_ratio,min_window_size_ratio,max_window_size_ratio):
        if window.shape[0] != win_height or window.shape[1] != win_width:
            continue
        prediction,prediction_probabilities,confidence=model.predict(window)
        #prediction = model.predict_window(window)
        #prediction_probabilities = model.predict_proba_window(window)
        #max_probabilities = np.max(prediction_probabilities, axis=1)
        
        if (prediction == 'frouge' or prediction == 'forange' or prediction == 'fvert') and confidence >= 0.99:  # Assuming '1' indicates a traffic sign
            x_orig = x
            y_orig = y
            w_orig = win_width
            h_orig = win_height
            prob = confidence
            detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, prediction, prob))
    
    
    print(detections)
    return detections


####
def selective_search(image):
    # 使用selectivesearch库进行选择搜索
    image_int = (image * 255).astype(np.uint8)
    img_lbl, regions = selectivesearch.selective_search(
        image_int, scale=500, sigma=0.7, min_size=100)
    
    candidates = set()
    for r in regions:
       
        if r['rect'] in candidates:
            continue
     
        if r['size'] < 1000:
            continue
        x, y, w, h = r['rect']
       
        if w == 0 or h == 0 or w / h > 1.3 or h / w > 3:
            continue
        candidates.add(r['rect'])
    """
    # 遍历候选区域并绘制矩形框
    for (x, y, w, h) in candidates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 显示结果图片
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    """
    return candidates

def detect_traffic_signs_with_selective_search(image, model):
    detections = []
    candidates = selective_search(image)
    
    for (x, y, w, h) in candidates:
        window = image[y:y + h, x:x + w]
        if window.shape[0] != h or window.shape[1] != w:
            continue
        prediction,prediction_probabilities,confidence=model.predict(window)
        
        
        prob = confidence
        if prediction != 'none' and prediction != 'frouge' and prediction != 'forange' and prediction != 'fvert' and prob > 0.9:
            detections.append((x, y, x + w, y + h, prediction, prob))
        elif (prediction == 'frouge' or prediction == 'forange' or prediction == 'fvert') and prob > 0.9:
            detections.append((x, y, x + w, y + h, prediction, prob))
    
    print(detections)
    return detections

def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    
    x1 = boxes[:, 0].astype(int)
    y1 = boxes[:, 1].astype(int)
    x2 = boxes[:, 2].astype(int)
    y2 = boxes[:, 3].astype(int)
    probs = boxes[:, 5].astype(float)
    
    idxs = np.argsort(probs)[::-1]
    
    picked_boxes = []
    
    while len(idxs) > 0:
        i = idxs[0]
        picked_boxes.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / ((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1))
        
        inside = (x1[idxs[1:]] >= x1[i]) & (y1[idxs[1:]] >= y1[i]) & (x2[idxs[1:]] <= x2[i]) & (y2[idxs[1:]] <= y2[i])
       
      
        idxs = np.delete(idxs, np.concatenate(([0], np.where((overlap > overlap_thresh) | inside)[0] + 1)))
        
        
    
    return np.array(picked_boxes)



#%%detection one images
"""
image = cv2.imread('C:/Users/23158/Desktop/SY32PROJET/dataset2/train/images/0833.jpg')
detections = detect_traffic_signs_with_selective_search(image, clf)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0.4)
#picked_boxes=boxes
print(picked_boxes)
for (x1, y1, x2, y2, label,max_probabilities) in picked_boxes:  
    x1, y1, x2, y2 ,max_probabilities= int(x1), int(y1), int(x2), int(y2),float(max_probabilities)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    text = f"{label}: {max_probabilities:.2f}"
    # 显示文本
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #output_image_path = os.path.join(output_folder, filename)
    #cv2.imwrite(output_image_path, image)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
plt.title('Detections')  
plt.axis('off')  
plt.show()
"""
#%%detection_images_in_folder
import pandas as pd
def detection_images_in_folder(folder_path, model, csv_path):
    results = []
    output_folder = os.path.join(os.getcwd(), 'output_images')
    
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            print(filename)
            image = cv2.imread(image_path)
            detections = detect_traffic_signs_with_selective_search(image, model)
            boxes = np.array(detections)
            picked_boxes = non_max_suppression(boxes, 0.1)  

            for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:
                x1, y1, x2, y2, max_probabilities = int(x1), int(y1), int(x2), int(y2), float(max_probabilities)
                results.append([filename, x1, y1, x2, y2, max_probabilities, label])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {max_probabilities:.2f}"
               
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            
            output_image_path = os.path.join(output_folder, filename)
            plt.figure()
            plt.imshow(image)
            
            cv2.imwrite(output_image_path, image)
         
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)

#detection_images_in_folder('C:/Users/23158/Desktop/SY32PROJET/dataset2/test', clf, 'detections.csv')

#%%
"""
from detection import build_detection_validation
label_path = "C:/Users/23158/Desktop/SY32PROJET/dataset2/train/labels"
build_detection_validation(label_path)
validations = pd.read_csv('validations.csv', header=None)
detections = pd.read_csv('detections.csv', header=None)
negative_samples_dir = "C:/Users/23158/Desktop/SY32PROJET/dataset2/train/cropped_images"
image_folder_path = "C:/Users/23158/Desktop/SY32PROJET/dataset2/train/images"

def iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    
    x_left = max(int(box1[0]), int(box2[0]))
    y_top = max(int(box1[1]), int(box2[1]))
    x_right = min(int(box1[2]), int(box2[2]))
    y_bottom = min(int(box1[3]), int(box2[3]))

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both the prediction and ground-truth rectangles
    box1_area = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
    box2_area = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))

    # The area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou

def get_negative_prediction(detections, validations, image_folder_path, negative_samples_dir):
    if image_folder_path== "dataset/val/images" or image_folder_path== "dataset/test":
        print("Cannot generate negative samples for validation or test set")
        return
    if not os.path.exists(negative_samples_dir):
        os.makedirs(negative_samples_dir)

    negative_predictions = []
    for _, detection_row in detections.iterrows():
        image_id_d, x_min_d, y_min_d, x_max_d, y_max_d, score_d, label_d = detection_row
        no_intersection = True

        for _, validation_row in validations.iterrows():
            image_id_v, x_min_v, y_min_v, x_max_v, y_max_v, label_v = validation_row

            if image_id_v == image_id_d:
                iou_value = iou((x_min_v, y_min_v, x_max_v, y_max_v), (x_min_d, y_min_d, x_max_d, y_max_d))
                if iou_value >= 0.3 or label_v == label_d:
                    no_intersection = False
                    break

        if no_intersection and label_d != 'none':
            negative_predictions.append((image_id_d, x_min_d, y_min_d, x_max_d, y_max_d, label_d))

    for i, (image_id, x_min, y_min, x_max, y_max, label) in enumerate(negative_predictions):
        print(image_id)
        image_path = os.path.join(image_folder_path, f"{image_id}")
        image = Image.open(image_path)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        save_path = os.path.join(negative_samples_dir, f"negative_sample_none_{i}.png")
        cropped_image.save(save_path)

get_negative_prediction(detections, validations,image_folder_path, negative_samples_dir)
"""