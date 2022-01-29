# Yoga assistant using human pose detection in machine learning

This virtual yoga instructor tells you if you are doing the right pose.

We use Body Pose Tracking with MediaPipe BlazePose for detecting human poses and classify yoga poses using sklearn SVC.

![yoga assistant using human pose detection in machine learning](https://i.ibb.co/Hr2pSFX/yoga-assistant.png)

You can download the dataset from Kaggle through this [link](https://www.kaggle.com/niharika41298/yoga-poses-dataset) . 

The dataset consists of 5 yoga poses. 

The core steps of this project were: 
- Building the dataset using BlazePose and Pandas methods
- Building the model using sklearn SVM in one-versus-one
- Evaluation of the model
- Predicting the pose for one image 
- Predicting the pose in a video and in real-time

Video demo here:

https://user-images.githubusercontent.com/46407601/151674413-dceddca6-0432-4ed7-af94-b2fef78da846.mp4



A further step would be using OpenPose or PoseNet instead to support real-time multi-person pose estimation. 
As BlazePose doesn't.
Another way is using YOLO for person detection, crop the persons then predict every cropped image.
The latter is a bit expensive in computational power. 

Anyway this would become a virtual yoga class instructor.

My initial plan was to build a complete fitness assistant. I still plan to do it and maybe this part in it.

Connect with me on [LinkedIN](https://www.linkedin.com/in/p%C3%A9rel-degla-b7b944138/) 
