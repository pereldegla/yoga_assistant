from sklearn.svm import SVC
from utils import *

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

data_train = data_train.drop(labels="Unnamed: 0", axis=1)  # delete if error about it
X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
model = SVC(kernel='poly', decision_function_shape='ovo')
model.fit(X, Y)
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=True, min_detection_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

# Test phase : build test dataset then evaluate
data_test.drop(labels="Unnamed: 0", axis=1, inplace=True)  # delete if error about it
predictions = evaluate(data_test, model, show=True)
# I think downdog and plank get these results because of the absence of variations in the hands positions
# Unlike tree and goddess which present some hand position variations

predict_video(model, "vid2.mp4", show=True)
#cv2.destroyAllWindows()
