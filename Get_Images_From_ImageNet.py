import tarfile
from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2 #long install, will do later (https://www.quora.com/How-do-I-install-Open-CV2-for-Python-3-6-in-Windows)
import PIL.Image
import urllib

import errno    
import os

from os import listdir
from os.path import isfile, join

from PIL import Image 

#from: https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python#600612
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

#from: https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5
page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289")#ship synset
#print(page.content) #commented out, long print
# BeautifulSoup is an HTML parsing library
soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line

str_soup=str(soup)#convert soup to string so it can be split
type(str_soup)
split_urls=str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(split_urls))#print the length of the list so you know how many urls you have

bikes_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02834778")#bicycle synset
#print(bikes_page.content) #commented out, long print
# BeautifulSoup is an HTML parsing library
from bs4 import BeautifulSoup
bikes_soup = BeautifulSoup(bikes_page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line

bikes_str_soup=str(bikes_soup)#convert soup to string so it can be split
type(bikes_str_soup)
bikes_split_urls=bikes_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(bikes_split_urls))

#!mkdir /content/train #create the Train folder
mkdir_p("./ImageNet/content/train") #create the Train folder
#!mkdir /content/train/ships #create the ships folder
mkdir_p("./ImageNet/content/train/ships") #create the ships folder
#!mkdir /content/train/bikes #create the bikes folder
mkdir_p("./ImageNet/content/train/bikes") #create the bikes folder
#!mkdir /content/validation
mkdir_p("./ImageNet/content/validation")
#!mkdir /content/validation/ships #create the ships folder
mkdir_p("./ImageNet/content/validation/ships") #create the ships folder
#!mkdir /content/validation/bikes #create the bikes folder
mkdir_p("./ImageNet/content/validation/bikes") #create the bikes folder

img_rows, img_cols = 32, 32 #number of rows and columns to convert the images to
input_shape = (img_rows, img_cols, 3)#format to store the images (rows, columns, channels) called channels last
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image
n_of_training_images=100#the number of training images to use
for progress in range(n_of_training_images):#store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print("Progress (ships):", progress)
    if not split_urls[progress] == None:
        try:
            I = url_to_image(split_urls[progress])
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = './ImageNet/content/train/ships/img'+str(progress)+'.jpg'#create a name of each image
                cv2.imwrite(save_path,I)
        except:
            None
#do the same for bikes:
for progress in range(n_of_training_images):#store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print("Progress (bikes):", progress)
    if not bikes_split_urls[progress] == None:
        try:
            I = url_to_image(bikes_split_urls[progress])
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = './ImageNet/content/train/bikes/img'+str(progress)+'.jpg'#create a name of each image
                cv2.imwrite(save_path,I)
        except:
            None
        
        
#Validation data:
for progress in range(50):#store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print("Progress (ships):", progress)
    if not split_urls[progress] == None:
        try:
            I = url_to_image(split_urls[n_of_training_images+progress])#get images that are different from the ones used for training
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = './ImageNet/content/validation/ships/img'+str(progress)+'.jpg'#create a name of each image
                cv2.imwrite(save_path,I)
        except:
            None
#do the same for bikes:
for progress in range(50):#store all the images on a directory
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print("Progress (bikes):", progress)
    if not bikes_split_urls[progress] == None:
        try:
            I = url_to_image(bikes_split_urls[n_of_training_images+progress])#get images that are different from the ones used for training
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = './ImageNet/content/validation/bikes/img'+str(progress)+'.jpg'#create a name of each image
                cv2.imwrite(save_path,I)
        except:
            None
        
print("\nTRAIN:\n")          
#print("\nlist the files inside ships directory:\n")        
#!ls /content/train/ships #list the files inside ships
mypath = "./ImageNet/content/train/ships"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("%s: "%(mypath), onlyfiles)
#print("\nlist the files inside bikes directory:\n")
#!ls /content/train/bikes #list the files inside bikes
mypath = "./ImageNet/content/train/bikes"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("%s: "%(mypath), onlyfiles)
print("\nVALIDATION:\n")
#print("\nlist the files inside ships directory:\n")        
#!ls /content/validation/ships #list the files inside ships
mypath = "./ImageNet/content/validation/ships"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("%s: "%(mypath), onlyfiles)
#print("\nlist the files inside bikes directory:\n")
#!ls /content/validation/bikes #list the files inside bikes
mypath = "./ImageNet/content/validation/bikes"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("%s: "%(mypath), onlyfiles)

img = Image.open("./ImageNet/content/validation/bikes/" + onlyfiles[0])
img.show()

#from: http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list
#first entry for a future test: n02119789

#from: https://stackoverflow.com/questions/15859089/get-the-directory-structure-of-a-tgz-file-in-python#15859216
#this is not yet accomplishing my goal (to read all ImageNet url entries)
##tar = tarfile.open("ImageNet/imagenet_fall11_urls.tgz")
##i = 0
##for file in tar.getmembers():
##    print(file.name) #print "fall11_urls.txt"
##    nested_file = tarfile.open(file.name)
##    #^ERROR: FileNotFoundError: [Errno 2] No such file or directory: 'fall11_urls.txt'
##    for url in nested_file.getmembers():
##        print(url.name)
##        if i > 3:
##            brake
##        i += 1
