# Extracting the Parameter Free Threshold Adjacency Statistics (pftas) from some
# image patches.
# This is my first python script, bear with me.

# ARgh !!! Remember always that python starts at 0. 0!

# Start by importing some packages. Before importing, the packages (modules?)
# needs to be installed. This is done in the Terminal: pip3 install imageio.

import imageio              # Read images.
import visvis
import mahotas              # The pftas function
import numpy as np          # General numerical/matrix manipulation
import pathlib              # Generating paths to where the images are
import time                 # tic-toc
import scipy.io as sio      # Save a python array to matlab (I know i shouldn't)
import sys                  # To exit

# Extracting random patches
from sklearn.feature_extraction.image import extract_patches_2d

## Testing it first on one image

# Reading a benign image
im = imageio.imread('fold1/test/400X/SOB_B_A-14-22549G-400-001.png')
# visvis.imshow(im)     # Having problems with this one.
print(im.shape)       # Check if it has the expected dimension.

patch_size = (64, 64) # Define patch size
patch_nr = 1000          # The number of patches extraced from each image
# Patch extraction
B = extract_patches_2d(im, patch_size, max_patches=patch_nr,random_state=0)

print(B.shape)
# input("Press Enter to continue...")

# Feature extraction
resim = mahotas.features.pftas(B[0,:,:,:])
print(resim.shape)              # Check if I got what I wanted - no idea
# print(resim)
# sys.exit()

# Generating paths to where my images are.
train_dir = pathlib.Path("/Users/kam025/ptn/fold4/train/400X/")
test_dir = pathlib.Path("/Users/kam025/ptn/fold4/test/400X/")

train_im = train_dir.rglob("*.png")
test_im = test_dir.rglob("*.png")

ltrain_im = list(train_im) # List all images in subdirs named 40
ltest_im = list(test_im) # List all images in subdirs named 40

mtrain_im = np.asarray(ltrain_im)

with open("fold4_400_train.txt", "w") as output:
    output.write(str(ltrain_im))

with open("fold4_400_test.txt", "w") as output:
    output.write(str(ltest_im))

print(len(ltrain_im))     # The number of images
print(len(ltest_im))     # The number of images
# input("Press Enter to continue...")  # Pausing to check if it's alright

# I will now create a double for-loop, so that I can extract the pftas from each
# patch in each image.

# Pre-allocate the output
pftas_train = np.zeros((len(resim), len(ltrain_im), patch_nr))
tic = time.clock()                        # Start timer
for i, im in enumerate(ltrain_im):      # enumerate gives index i and value im
  # Extract patches for each image
  print(i)
  B = extract_patches_2d(imageio.imread(im), patch_size, max_patches=patch_nr, random_state=0)
  for j in range(0,patch_nr):
    # pftas for each patch
    pftas_train[:,i,j] = mahotas.features.pftas(B[j,:,:,:])

toc = time.clock()                        # Stop timer
print(toc - tic)

print(B.shape)
print(pftas_train.shape)
# print(pftas_ben[160,0,:])                     # For comparison with my .mat file
# print(pftas_ben[160,1,:])                     # For comparison with my .mat file

# Reshape to 2d
pftas_train = np.reshape(pftas_train, (len(resim), patch_nr*len(ltrain_im)))
# print(pftas_ben[160,0:20])

sio.savemat('pftas_train_fold4_400.mat', dict(pftas_train = pftas_train)) # Save

# input("Press Enter to continue...")

# Repeat for the test images
pftas_test = np.zeros((len(resim), len(ltest_im), patch_nr))
tic = time.clock()                        # Start timer
for i, im in enumerate(ltest_im):      # enumerate gives index i and value im
  # Extract patches for each image
  B = extract_patches_2d(imageio.imread(im), patch_size, max_patches=patch_nr, random_state=0)
  for j in range(0,patch_nr):
    # pftas for each patch
    pftas_test[:,i,j] = mahotas.features.pftas(B[j,:,:,:])

toc = time.clock()                        # Stop timer
print(toc - tic)

# print(pftas_mal[160,0,:])                     # For comparison with my .mat file
# Reshape to 2d
pftas_test = np.reshape(pftas_test, (len(resim), patch_nr*len(ltest_im)))

sio.savemat('pftas_test_fold4_400.mat', dict(pftas_test = pftas_test)) # Save
