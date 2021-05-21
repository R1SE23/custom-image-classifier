from botnoi import scrape as sc
from botnoi import cv
import os
import glob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import time


# trained classifier model based on collected dataset
DATADIR = '/content/drive/MyDrive/Dataset/'
CATEGORIES = ['cardboard',
           'glass',
           'metal',
           'paper',
           'plastic',
           'trash']

# extract feature of images
def extractimagefeat(query):
  foldername = 'images/'+query
  isdir = os.path.isdir(foldername) 
  if not isdir:
    os.makedirs(foldername)
  print(foldername)
  i = 1
  path = os.path.join(DATADIR, query)
  for img in os.listdir(path):
    print(img)
    #extract image features from each images and save to files
    savepath = foldername + '/' + str(i)+'.p'
    a = cv.image(path + '/' + img)
    a.getresnet50()
    a.save(savepath)
    i = i + 1
    
  return 'complete'


def createdataset():
  imgfolder = glob.glob('images/*')
  dataset = []
  for cls in imgfolder:
    clsset = pd.DataFrame()
    pList = glob.glob(cls+'/*')
    featvec = []
    for p in pList:
      dat = pickle.load(open(p,'rb'))
      #featvec.append(dat.restnet50)
      featvec.append(dat.resnet50)

    clsset['feature'] = featvec
    cls = cls.split('/')[-1]
    clsset['label'] = cls
    dataset.append(clsset)
  return pd.concat(dataset,axis=0)



# train model
def trainmodel(dataset,modfile=''):
  trainfeat, testfeat, trainlabel, testlabel = train_test_split(dataset['feature'], dataset['label'], test_size=0.20, random_state=42)
  clf = LinearSVC()
  clf = CalibratedClassifierCV(clf) 
  mod = clf.fit(np.vstack(trainfeat.values),trainlabel.values)
  res = mod.predict(np.vstack(testfeat.values))
  if modfile!='':
    pickle.dump(mod,open(modfile,'wb'))
  acc = sum(res == testlabel.values)/len(res)
  return mod,acc


# predict image from image address url
def predicting(imgurl):
  modFile = 'waste-classifier-model.pkl'
  mod = pickle.load(open(modFile, 'rb'))
  # read file
  a = cv.image(imgurl)
  feat = a.getresnet50()
  probList = mod.predict_proba([feat])[0]
  maxprobind = np.argmax(probList)
  prob = probList[maxprobind]
  outclass = mod.classes_[maxprobind]
  result = {}
  result['class'] = outclass
  result['probability'] = prob
  return result


def run():
  start = time.time()
  for category in CATEGORIES:
      extractimagefeat(category)
  dataset = createdataset()
  modFile = 'waste-classifier-model.pkl'
  mod, acc = trainmodel(dataset, modFile)
  end = time.time()
  print(end - start)
  return predicting('https://m.media-amazon.com/images/I/61OorFhm6SL._AC_SX569_.jpg')