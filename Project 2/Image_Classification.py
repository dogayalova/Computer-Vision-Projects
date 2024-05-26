"""We will first:
1- Prepare Data
2- Train / Test split
3- Train classifier
4- Test performance
 We will use scikit-learn (image classifier comes from this), scikit-image and numpy. 
"""

import os 
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split #for train/test split
from sklearn.model_selection import GridSearchCV #for train classifier
from sklearn.svm import SVC #for train classifier
from sklearn.metrics import accuracy_score #for test performance


#Prepare Data
input_dir = '/Users/dogayalova/Desktop/clf-data'
categories = ["empty", "not_empty"]

data = []
labels = [] 
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img,(15, 15))
        data.append(img.flatten()) # Make the image unidimentional which is in the form of a matrix in 15x15. We want to convert it into an array, a string.
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)



#Train / Test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle =True, stratify= labels) #We split the whole data to train and test sets. "test_size" indicates %20 of data is for test. Get rid of bias with "shuffle". "Stratify" keeps the same proportions of the labels. 

#Train classifier
classifier = SVC()

parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}] #we are going to train 3x4=12 image classifiers, as gama has 3 values and c has 4 values.

grid_search = GridSearchCV(classifier, parameters) #to train as many classifiers at once 

grid_search.fit(x_train, y_train)

#Test performance
best_estimator = grid_search.best_estimator_ #this is our model, our classifier!

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of sampels were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open("./model.p", "wb"))