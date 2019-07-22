from django.shortcuts import render, get_object_or_404, redirect
from .models import User, UserProfile, Position, Candidate
from django.http import HttpResponseRedirect
from django.contrib.auth import logout, login, authenticate
from django.contrib.auth.decorators import login_required
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
import base64
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc
from PIL import Image
from django.conf import settings
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

def index(request):
    return render(request, 'Vote/home.html')

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return HttpResponseRedirect('/Vote/detect/')
            else:
                return HttpResponseRedirect("Account disabled")
        else:
            print("Invalid credentials: {0}, {1}".format(username, password))
            return HttpResponseRedirect('/Vote/invalid/')
    else:
        return render(request, 'Vote/login.html', {})


@login_required(redirect_field_name='/Vote/login/')
def user_logout(request):
    logout(request)
    return HttpResponseRedirect('/Vote/home/')


def home(request):
    context = {}
    return render(request, 'Vote/home.html', context)


def home_hindi(request):
    context = {}
    return render(request, 'Vote/home_hindi.html', context)

def vote(request):
    context = {}
    try:
        pos = Position.objects.all()
        user = User.objects.get(username=request.user.username)
        profile = UserProfile.objects.get(user=user)
        context['candidates'] = []
        if profile.voted:
            return HttpResponseRedirect('/Vote/voted/')
        else:
            for c in pos:
                can = []
                candidate = Candidate.objects.filter(candidate=c)
                for i in range(0, c.no_of_candidates):
                    can.append([candidate[i], c.position])
                context['candidates'].append(can)
    except:
        return HttpResponseRedirect("/Vote/login/")
    if request.method == 'POST':
        pos = Position.objects.all()
        for c in pos:
            s = 'candidate' + c.position
            selected_candidate = Candidate.objects.get(pk=request.POST[s])
            selected_candidate.votes += 1
            selected_candidate.save()
            profile.voted = True
            profile.save()
            return render(request, 'Vote/casted.html', context)
        else:
            print("No Post")

    return render(request, 'Vote/vote.html', context)


def results(request):
    context = {}
    pos = Position.objects.all()
    context['candidates'] = []
    for c in pos:
        fig = plt.figure()
        vot = []   # No of votes for graph
        can = []
        cand = []  # Name of the candidates for graph labelling
        candidate = Candidate.objects.filter(candidate=c)
        for i in range(0, c.no_of_candidates):
            can.append([candidate[i]])
            cand.append(candidate[i].name)
            vot.append(candidate[i].votes)

        plt.pie(vot, labels=cand, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        name = "/home/anurag/Desktop/Smart-Voting-System/Vote/static/Vote/" + c.position + '.png'
        fig.savefig(name)
        can[0].append(c)
        can[0].append(name)
        plt.close(fig)
        context['candidates'].append(can)

    return render(request, 'Vote/results.html', context)


def about(request):
    context = {}
    return render(request, 'Vote/about.html', context)


def voted(request):
    context = {}
    return render(request, 'Vote/voted.html', context)

def invalid(request):
    context = {}
    return render(request, 'Vote/invalid.html', context)

def casted(request):
    context = {}
    return render(request, 'Vote/casted.html', context)

def create_dataset(request):
    #print request.POST
    userId = request.POST['userId']
    print (cv2.__version__)
    # Detect face
    #Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier('/home/anurag/Desktop/Smart-Voting-System/ml/haarcascade_frontalface_default.xml')
    #camture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = userId
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        #cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        #the returned img is a colored image but for the classifier to work we need a greyscale image
        #to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for(x,y,w,h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite('/home/anurag/Desktop/Smart-Voting-System/ml/dataset/.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Face",img)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        if(sampleNum>35):
            break
    #releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

    return redirect('/Vote/face_index')

def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    from PIL import Image

    #Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path = '/home/anurag/Desktop/Smart-Voting-System/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        #print imagePaths

        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L') #convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format
            ID = int(os.path.split(imagePath)[-1].split('.')[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            #print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    #Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save('/home/anurag/Desktop/Smart-Voting-System/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/Vote/home')


def detect(request):
    faceDetect = cv2.CascadeClassifier('/home/anurag/Desktop/Smart-Voting-System/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read('/home/anurag/Desktop/Smart-Voting-System/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            getId,conf = rec.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

            #print conf;
            if conf<35:
                userId = getId
                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
            else:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)

            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke

        cv2.imshow("Face",img)
        if(cv2.waitKey(1) == ord('q')):
            break
        elif(userId != 0):
            cv2.waitKey(1000)
            cam.release()
            cv2.destroyAllWindows()
            return HttpResponseRedirect('/records/details/'+str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/Vote/home')

def face_index(request):
    return render(request, 'face_index.html')
