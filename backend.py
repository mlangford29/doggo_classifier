# let's get this backend stuff started!

# imports
import os
import sys
import urllib.request, json
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, session, render_template, request

from src.common import consts
from src.data_preparation import dataset
from src.freezing import freeze
from src.common import paths

app = Flask(__name__)

# just the root!
@app.route('/')
def root():
	print(os.getcwd())
	return render_template('index.html')

# function that will take the image and hit the model
# not sure if this picture needs to be in the byte form but we'll check this out later
##### CURRENTLY A PIC PATH BUT WOULD BE BETTER TO NOT BE
@app.route('/hit_model')
def hit_model(pic_path):



	def infer(model_name, img_raw):
	    with tf.Graph().as_default(), tf.Session().as_default() as sess:
	        tensors = freeze.unfreeze_into_current_graph(
	            os.path.join(paths.FROZEN_MODELS_DIR, model_name + '.pb'),
	            tensor_names=[consts.INCEPTION_INPUT_TENSOR, consts.OUTPUT_TENSOR_NAME])

	        _, one_hot_decoder = dataset.one_hot_label_encoder()

	        probs = sess.run(tensors[consts.OUTPUT_TENSOR_NAME],
	                         feed_dict={tensors[consts.INCEPTION_INPUT_TENSOR]: img_raw})

	        breeds = one_hot_decoder(np.identity(consts.CLASSES_COUNT)).reshape(-1)

	        df = pd.DataFrame(data={'prob': probs.reshape(-1), 'breed': breeds})


	        return df.sort_values(['prob'], ascending=False)

	##### IF YOU CAN TAKE IN RAW CAMERA FEED, THAT WOULD BE GREAT
	with open(pic_path, 'rb') as f:
		img_raw = f.read()

	'''
	def classify(resource_type, path):
	    
	    if resource_type == 'uri':
	        response = urllib.urlopen(path)
	        img_raw = response.read()
	    else:

	        with open(path, 'rb') as f:
	'''
	
	#return infer(consts.CURRENT_MODEL_NAME, img_raw)
	probs = infer(consts.CURRENT_MODEL_NAME, img_raw)
	
	'''
	if __name__ == '__main__':
	    src = sys.argv[1]
	    path = sys.argv[2] # uri to a dog image to classify
	    probs = classify(src, path)

	    print(probs.sort_values(['prob'], ascending=False).take(range(5)))
	'''

	# returning the top 5 of them, sorted!
	return probs.sort_values(['prob'], ascending=False).take(range(5))


# function that checks to see if we're below confidence threshold
# with the highest dog

# hit the petfinder API!
##### HERE IS AN EXAMPLE REQUEST
# http://api.petfinder.com/pet.find?format=json&key=73b809ff630a0a072606a63a533503fe&animal=dog&breed=Golden%20Retriever&count=1&output=full&location=78705
@app.route('/hit_petfinder')
def hit_petfinder(breed, count = 1):

	# first, we need to develop that url
	### CHECK TO SEE IF THERE'S A SPACE IN DOG BREED NAME
	if ' ' in breed:
		breed = breed.replace(' ', '%20')

	##### THIS IS WHERE WE WOULD GET THE LOCATION
	##### BUT RIGHT NOW I'M HARDCODING IT TO 78705
	loc = '78705'

	url = 'http://api.petfinder.com/pet.find?format=json&key=73b809ff630a0a072606a63a533503fe&animal=dog&breed={}&count={}&output=full&location={}'.format(
			breed, count, loc)

	with urllib.request.urlopen(url) as r:
		global dog_data
		dog_data = json.loads(r.read().decode())

# function that converts lower_case dog form into a fancier one
#@app.route('/convert_breed')
def convert_dog_breed(breed):

	# if there's an underscore, replace with a space
	breed_tokens = breed.split('_')

	breed_str = ''

	# then capitalize all words in the string
	for breed_tok in breed_tokens:

		if breed_str != '':
			breed_str += ' '
		breed_str += breed_tok.capitalize()

	return breed_str

# takes json data and returns the name of the dog
def find_name():

	##### these are all using global data for the dog data!
	return dog_data['petfinder']['pets']['pet']['name']['$t']

# takes json data and returns the breed(s) of the dog
##### you need to check "mix" to see if that's yes

# takes json data and returns the age of the dog
def find_age():
	return dog_data['petfinder']['pets']['pet']['age']['$t']

# takes json data and returns the sex of the dog
def find_sex():
	return dog_data['petfinder']['pets']['pet']['sex']['$t']

# takes json data and returns URL of image of the dog
# ok actually there may be a whole bunch of urls, so we'll get them in a list
def find_photo_url_list():
	url_list = []
	for photo_dict in dog_data['petfinder']['pets']['pet']['media']['photos']['photo']:
		print(photo_dict)
		url_list.append(photo_dict['$t'])
	return url_list

# takes json data and returns string of the description of the dog
def find_description():
	return dog_data['petfinder']['pets']['pet']['description']['$t']


# main for right now
if __name__ == '__main__':

	#app = Flask('main') ##### not sure if the name matters here
	app.run(debug=True)

	'''
	probs = hit_model('golden.jpeg')
	print(probs)
	breed_name = convert_dog_breed(probs.iloc[0]['breed'])

	##### this will create the global json for dog_data
	hit_petfinder(breed_name)
	'''

	







