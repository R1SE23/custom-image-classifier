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

def extractimagefeat(query):
  foldername = 'images/'+query
  isdir = os.path.isdir(foldername) 
  if not isdir:
    os.makedirs(foldername)
  #get images from google search
  imglist = sc.get_image_urls(query)
  i = 1
  # specify image num
  image_num = 50
  for img in imglist[0:image_num]:
    #extract image features from each images and save to files
    try:
      print(i)
      savepath = foldername + '/' + str(i)+'.p'
      a = cv.image(img)
      a.getresnet50()
      a.save(savepath)
      i = i + 1
    except:
      pass
  return 'complete'

def createdataset(imgfolder):
  #imgfolder = glob.glob('images/*')
  dataset = []
  for cls in imgfolder:
    clsset = pd.DataFrame()
    pList = glob.glob(cls+'/*')
    featvec = []
    for p in pList:
      dat = pickle.load(open(p,'rb'))
      featvec.append(dat.resnet50)
    #featvec = np.vstack(featvec)
    clsset['feature'] = featvec
    cls = cls.split('/')[-1]
    clsset['label'] = cls
    dataset.append(clsset)
  return pd.concat(dataset,axis=0)


def trainmodel(dataset, modFile=''):
  basepath = os.path.dirname(os.path.abspath("__file__"))
  trainfeat, testfeat, trainlabel, testlabel = train_test_split(dataset['feature'], dataset['label'], test_size=0.20, random_state=42)
  clf = LinearSVC()
  clf = CalibratedClassifierCV(clf) 
  mod = clf.fit(np.vstack(trainfeat.values),trainlabel.values)
  res = mod.predict(np.vstack(testfeat.values))
  if modFile!='':
    file_path = os.path.join(basepath, 'classifier_model')
    print(file_path)
    with open(f'{file_path}/{modFile}', 'wb') as f:
      pickle.dump(mod, f)
  acc = sum(res == testlabel.values)/len(res)
  return mod, acc


def predicting(imgurl, mod):
  # modFile = 'vehicle-classification.p'
  # mod = pickle.load(open(modFile, 'rb'))

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

def run(object_to_classify):
    name = ""
    for cls in object_to_classify:
      if object_to_classify[-1] != cls:
        name += cls + "-"
      else:
        name += cls
    
    start = time.time()
    # image class to train classifier
    # when run locally. specify a class list like below then all run() function
    ## clsList = ['wagons', 'bicycles', 'motor vehicles', 'railed vehicles', 'watercraft', 'amphibious vehicles', 'aircraft', 'spacecraft']
    for c in object_to_classify:
        extractimagefeat(c)
    imgfolder = ['images/'+c for c in object_to_classify]
    dataset = createdataset(imgfolder)
    # name of classifier
    modFile = name + ".p"
    mod, acc = trainmodel(dataset, modFile)
    
    end = time.time()
    print(end - start)

    # predict some image
    predicting('https://i.pinimg.com/originals/79/dd/6c/79dd6c94ce2a75886878e4cef199e9e9.jpg', mod)
    
    return acc, modFile