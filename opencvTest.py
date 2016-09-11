import cv2
import sys
import time
import glob
import random
import numpy as np
import sched
import requests
import json
from gtts import gTTS
import os

cascPath = sys.argv[1]
eyePath = sys.argv[2]

count = 0
total_time = 0
yawning = 0
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list

faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
fishface = cv2.face.createFisherFaceRecognizer();
s = sched.scheduler(time.time, time.sleep)


def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    #prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training 

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale    
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
        '''
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
        '''
        
    return training_data, training_labels

def execute():
    global count
    global total_time
    headers = {'Content-Type': 'application/json'}
    url = 'http://52.24.68.163/mood'

    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        t_end = time.time() + 1;
        while time.time() < t_end:
            continue
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if faces is not():
            if total_time > 2400:
                content = 'I think you should take a small break'
                print(content);
                tts = gTTS(text=content, lang='en')
                tts.save("break.mp3")
                print("created file");
                os.startfile('break.mp3')
                total_time = 0
            else:
                total_time = total_time+1
                
        else:
            total_time = 0
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            gray = cv2.resize(gray, (350,350))
            pred = fishface.predict(gray)
            #pred = 0
            if pred == 1 or pred == 3 or pred == 4 or pred == 6:
                data = {"mood":"need_help"}
                r = requests.post(url, data=json.dumps(data), headers=headers)
                print(emotions[pred]);
            elif pred == 0:
                eyes = eyeCascade.detectMultiScale(roi_gray)
                if eyes is not():
                    print("opened");
                    count = 0
                else:
                    print("closed");
                    count = count + 1
                    if count > 10:
                            content = 'Coffee Time!'
                            print(content);
                            tts = gTTS(text=content, lang='en')
                            tts.save("sleepy.mp3")
                            print("created file");
                            os.startfile('sleepy.mp3')
                            count = 0
                #for (ex,ey,ew,eh) in eyes:
                #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2);
                     

        # Display the resulting frame
        cv2.imshow('Video', frame)

        #s.enter(300, 1, execute, ())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    training_data, training_labels = make_sets()
    print("training fisher face classifier");
    print("size of training set is:", len(training_labels), "images");
    print(training_data[0].shape);
    fishface.train(training_data, np.asarray(training_labels))
    print("Done training");
    execute()