from PIL import Image
import os
import numpy as np
from matplotlib.pyplot import imsave

categories = ['AKIEC/','BCC/','BKL/','DF/','MEL/','NV/','VASC/']

treshold = 200


for category in categories:

	x_mins = []
	x_maxs = []
	y_mins = []
	y_maxs = []

	img_sizes = []
	
	root = 'data/isic2017/test/images/'

	for myfile in os.listdir(root+category):
		pil_img = Image.open(root+category+myfile)
		img_sizes.append(pil_img.size)
		
	root = 'data/isic2017/test/masks/'

	for i in range(len(os.listdir(root+category))):
		filename = root+category+str(i)+'_predict.png'
		pil_img = Image.open(filename)
		pil_img = pil_img.resize(img_sizes[i])
		img = np.asarray(pil_img)
		img = img/255
		x_lesion = []
		for j in range(img.shape[0]):
			for k in range(img.shape[1]):
				if (img[j][k]>treshold):
					x_lesion.append(j)
					break
		y_lesion = []
		for j in range(img.shape[1]):
			for k in range(img.shape[0]):
				if (img[k][j]>treshold):
					y_lesion.append(j)
					break
		try:
			x_min = min(x_lesion)
			x_mins.append(x_min)
		except ValueError:
			x_mins.append(-1)
			x_maxs.append(-1)
			y_mins.append(-1)
			y_maxs.append(-1)
			continue
		try:
			x_max = max(x_lesion)
			x_maxs.append(x_max)
		except ValueError:
			x_maxs.append(-1)
			y_mins.append(-1)
			y_maxs.append(-1)
			continue
		try:
			y_min = min(y_lesion)
			y_mins.append(y_min)
		except ValueError:
			y_mins.append(-1)
			y_maxs.append(-1)
			continue
		try:
			y_max = max(y_lesion)
			y_maxs.append(y_max)
		except ValueError:
			y_maxs.append(-1)
			continue
		
	for i in range(len(os.listdir(root+category))):
		if x_mins[i] == -1:
			i+=1
			continue
		filename = root+category+str(i)+'_predict.png'
		pil_img = Image.open(filename)
		pil_img.resize(img_sizes[i])
		cropped_pil_img = pil_img.crop(
			(
				y_mins[i],
				x_mins[i],
				y_maxs[i],
				x_maxs[i]
			)
		)
		cropped_img = np.asarray(cropped_pil_img)
		imsave(root+'cropped/'+category+str(i)+'_predict.png', cropped_img)
			
	root = 'data/isic2017/test/images/'
	
	i=0
	for myfile in os.listdir(root+category):
		if x_mins[i] == -1:
			i+=1
			continue
		print(root+category+myfile)
		pil_img = Image.open(root+category+myfile)
		cropped_pil_img = pil_img.crop(
			(
				y_mins[i],
				x_mins[i],
				y_maxs[i],
				x_maxs[i]
			)
		)
		cropped_img = np.asarray(cropped_pil_img)
		imsave(root+'cropped/'+category+myfile, cropped_img)
		i+=1