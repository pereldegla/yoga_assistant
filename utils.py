## UNCOMMENT BOTTOM LINES AND RUN THIS ONCE
import mediapipe as mp
import cv2
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report

# Build the dataset using landmarks detection and save it as csv
def build_dataset(path, dataset_type):
        # path: ROOT PATH TO DATASET
        # dataset_type: type of dataset
        data = []
        for p in points:
                x = str(p)[13:]
                data.append(x + "_x")
                data.append(x + "_y")
                data.append(x + "_z")
                data.append(x + "_vis")
        data.append("target")  # name of the position
        data = pd.DataFrame(columns=data)  # Empty dataset
        count = 0

        dirnames = [x[1] for x in os.walk(path)][0]
        # walking through the whole training dataset
        for k in range(len(dirnames)):
                for img in os.listdir(path + "/" + dirnames[k]):
                        temp = []
                        img = cv2.imread(path + "/" + dirnames[k] + "/" +img)
                        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = pose.process(imgRGB)

                        if results.pose_landmarks:
                                landmarks = results.pose_landmarks.landmark

                                for i, j in zip(points, landmarks):

                                        temp = temp + [j.x, j.y, j.z, j.visibility]

                                temp.append(dirnames[k]) #adding pos_name to dataframe

                                data.loc[count] = temp
                                count +=1
        data.to_csv(dataset_type+".csv") # save the data_train as a csv file | viewing on ExcelReader might suck


# Predict the name of the poses in the image
def predict(img, model, show=False):
        temp = []
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for j in landmarks:
                        temp = temp + [j.x, j.y, j.z, j.visibility]
                y = model.predict([temp])

                if show:
                        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                        cv2.putText(img, str(y[0]), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)
                        cv2.imshow("image", img)
                        cv2.waitKey(0)


def predict_video(model, video="0", show=False):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
                temp = []
                success, img = cap.read()
                if not success:
                        print("Ignoring empty camera frame.")
                        continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img)
                if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        for j in landmarks:
                                temp = temp + [j.x, j.y, j.z, j.visibility]
                        y = model.predict([temp])
                        name = str(y[0])
                        if show:
                                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                                (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                                cv2.rectangle(img, (40, 40), (40+w, 60), (255, 255, 255), cv2.FILLED)
                                cv2.putText(img, name, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                                cv2.imshow("Video", img)
                                if cv2.waitKey(5) & 0xFF == 27:
                                        break
        cap.release()


# Use this to evaluate any dataset you've built
def evaluate(data_test, model, show=False):
        target = data_test.loc[:, "target"]  # list of labels
        target = target.values.tolist()
        predictions = []
        for i in range(len(data_test)):
                tmp = data_test.iloc[i, 0:len(data_test.columns) - 1]
                tmp = tmp.values.tolist()
                predictions.append(model.predict([tmp])[0])
        if show:
                print(confusion_matrix(predictions, target), '\n')
                print(classification_report(predictions, target))
        return predictions


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # For drawing keypoints
points = mpPose.PoseLandmark  # Landmarks
#build_dataset("DATASET/TRAIN", "train")
#build_dataset("DATASET/TEST", "test")




