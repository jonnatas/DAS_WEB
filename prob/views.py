from django.http import HttpResponse
from django.shortcuts import render, render_to_response
from django.template import loader, RequestContext
from .forms import NameForm
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
import matplotlib.pyplot as plt
import urllib
import h5py
import sys
import os
from scipy import misc
from IPython.display import clear_output # to clear command output when this notebook gets too cluttered

labels = []
images_path = 'images/'

#Caregar o caffe
#**********************************************************
#
home_dir = os.getenv("HOME")
caffe_root = os.getenv("CAFFE_ROOT")  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, os.path.join(caffe_root, 'python'))


import caffe

if os.path.isfile(os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system(caffe_root+"/scripts/download_model_binary.py ~/caffe/models/bvlc_reference_caffenet")


caffe.set_mode_cpu()

model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# load ImageNet labels
labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
if not os.path.exists(labels_file):
    os.system(caffe_root+"/data/ilsvrc12/get_ilsvrc_aux.sh")
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

# Function to process the query/test image
#**************************************************************************************'''

def predict_imageNet(image_filename):
    image = caffe.io.load_image(image_filename)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)

    # perform classification
    net.forward()

    # obtain the output probabilities
    output_prob = net.blobs['prob'].data[0]

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]

    plt.imshow(image)
    plt.axis('off')

    print 'probabilities and labels:'
    predictions = zip(output_prob[top_inds], labels[top_inds]) # showing only labels (skipping the index)
    for p in predictions:
        print p
    
    # plt.figure(figsize=(15, 3))
    # plt.plot(output_prob)
    return output_prob

# Function to pre-process (or load) image dataset
#*
def load_dataset(images_path):
    # Load/build a dataset of vectors (i.e. a big matrix) of probabilities
    # from the ImageNet ILSVRC 2012 challenge using Caffe.
    vectors_filename = os.path.join(images_path, 'vectors.h5')

    if os.path.exists(vectors_filename):
        print 'Loading image signatures (probability vectors) from ' + vectors_filename
        with h5py.File(vectors_filename, 'r') as f:
            vectors = f['vectors'][()]
            img_files = f['img_files'][()]

    else:
        # Build a list of JPG files (change if you want other image types):
        os.listdir(images_path)
        img_files = [f for f in os.listdir(images_path) if (('jpg' in f) or ('JPG') in f)]

        print 'Loading all images to the memory and pre-processing them...'
        
        net_data_shape = net.blobs['data'].data.shape
        train_images = np.zeros(([len(img_files)] + list(net_data_shape[1:])))

        for (f,n) in zip(img_files, range(len(img_files))):
            print '%d %s'% (n,f)
            image = caffe.io.load_image(os.path.join(images_path, f))
            train_images[n] = transformer.preprocess('data', image)
    
        print 'Extracting descriptor vector (classifying) for all images...'
        vectors = np.zeros((train_images.shape[0],1000))
        for n in range(0,train_images.shape[0],10): # For each batch of 10 images:
            # This block can/should be parallelised!
            print 'Processing batch %d' % n
            last_n = np.min((n+10, train_images.shape[0]))

            net.blobs['data'].data[0:last_n-n] = train_images[n:last_n]

            # perform classification
            net.forward()

            # obtain the output probabilities
            vectors[n:last_n] = net.blobs['prob'].data[0:last_n-n]
        
        print 'Saving descriptors and file indices to ' + vectors_filename
        with h5py.File(vectors_filename, 'w') as f:
            f.create_dataset('vectors', data=vectors)
            f.create_dataset('img_files', data=img_files)
    
    return vectors, img_files


#  Nearest neighbour class
#* ************************************************

class NearestNeighbors:
    def __init__(self, K=10, Xtr=[], images_path='Photos/', img_files=[], labels=np.empty(0)):
        # Setting defaults
        self.K = K
        self.Xtr = Xtr
        self.images_path = images_path
        self.img_files = img_files
        self.labels = labels

    def setXtr(self, Xtr):
        """ X is N x D where each row is an example."""
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = Xtr
        
    def setK(self, K):
        """ K is the number of samples to be retrieved for each query."""
        self.K = K

    def setImagesPath(self,images_path):
        self.images_path = images_path
        
    def setFilesList(self,img_files):
        self.img_files = img_files

    def setLabels(self,labels):
        self.labels = labels
        
    def predict(self, x):
        distances = np.sum(np.abs(self.Xtr-x), axis = 1)
        return np.argsort(distances) # returns an array of indices of of the samples, sorted by how similar they are to x.

    def retrieve(self, x):
        # The next 3 lines are for debugging purposes:
        nearest_neighbours = self.predict(x)
        images = []

        for n in range(self.K):
            idx = nearest_neighbours[n]
            
            images.append('/' + (os.path.join(self.images_path, self.img_files[idx])))


def index(request):

	template = loader.get_template('prob/index.html')

	context = {
        'latest_question_list': '',
    }

	return render(request,'prob/index.html')

def detail(request):

    my_image_url = request.POST['url']
    #fs = FileSystemStorage()
    uploaded_file_url = "image.jpg"

    urllib.urlretrieve (my_image_url, uploaded_file_url)
    #uploaded_file_url = fs.url(my_image_url)
    predictions = predict_imageNet(uploaded_file_url)

    template = loader.get_template('prob/detail.html')

    context = {
        'predictions': predictions,
    }
    return render(request,'prob/detail.html', context)