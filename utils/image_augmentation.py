
import random
import numpy as np
import cv2
import os 



seq = [None]

def load_aug():
	import imgaug as ia
	from imgaug import augmenters as iaa
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)
	seq[0] = iaa.Sequential([iaa.Fliplr(0.5), # horizontally flip 50% of all images
							iaa.Flipud(0.2), # vertically flip 20% of all images
							sometimes(iaa.Affine(
									scale=(0.95, 1.05), # scale images to 80-120% of their size, individually per axis
									translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
									rotate=(-45, 45)))],
							random_order=True)

def _augment_seg(img,seg):
	import imgaug as ia
	if seq[0] is None:
		load_aug()
	aug_det = seq[0].to_deterministic() 
	image_aug = aug_det.augment_image( img )
	segmap = ia.SegmentationMapOnImage( seg , nb_classes=np.max(seg)+1 , shape=img.shape )
	segmap_aug = aug_det.augment_segmentation_maps( segmap )
	segmap_aug = segmap_aug.get_arr_int()
	return image_aug , segmap_aug

def try_n_times( fn , n , *args , **kargs):
	attempts = 0
	while attempts < n:
		try:
			return fn( *args , **kargs )
		except Exception as e:
			attempts += 1
	return fn( *args , **kargs )

def augment_seg( img , seg  ):
	return try_n_times(_augment_seg,0,img,seg)

if __name__=="__main__":
	import imgaug as ia
	import matplotlib.pyplot as plt
	seq = [None]
	if seq[0] is None:
		load_aug()
	

	directory='E:/glass2/model/pythonModels/data/5/images_prepped_train'
	filenames=os.listdir(directory)
	for filename in filenames:
		imagepath=os.path.join(directory,filename)
		img=cv2.imdecode(np.fromfile(imagepath,dtype=np.uint8),-1)
		aug_det = seq[0].to_deterministic() 
		image_aug = aug_det.augment_image(img)
		plt.imshow(image_aug)
		plt.show()