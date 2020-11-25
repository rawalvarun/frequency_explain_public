
'''
python diff_DCT.py --first ../generic_pipeline/alexnet_finetune_BFGS/correct_DCT_targeted.png  --second ../generic_pipeline/densenet_BFGS/correct_DCT_targeted.png -a alexnet -b densenet

adding --all or not can lead to more or few diff points


python diff_DCT.py --first ../generic_pipeline/small_batch/densenet_PGD/logscale_correct_DCT_targeted.png --second ../generic_pipeline/small_batch/squeezenet_PGD/logscale_correct_DCT_targeted.png -a densenet -b squeezenet

'''

# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second")

ap.add_argument("-A", "--all", required=False,  action='store_true',
	help="show all differences or only the highest ones ")

ap.add_argument("-a", "--first-label", required=True,
	help="first input image legend label")
ap.add_argument("-b", "--second-label", required=True,
	help="second input image legend label")

args = vars(ap.parse_args())

if not os.path.exists(args["first"]):
	raise Exception("First file doesn't exist")

if not os.path.exists(args["second"]):
	raise Exception("Second file doesn't exist")

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
# convert the images to grayscale

# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

grayA = imageA
grayB = imageB

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(imageA, imageB, full=True,  multichannel=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


# # threshold the difference image, followed by finding contours to
# # obtain the regions of the two input images that differ
# thresh = cv2.threshold(diff, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)


# # loop over the contours
# for c in cnts:
# 	# compute the bounding box of the contour and then draw the
# 	# bounding box on both input images to represent where the two
# 	# images differ
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# # show the output images
# cv2.imwrite("Original", imageA)
# cv2.imwrite("Modified", imageB)
# cv2.imwrite("Diff", diff)
# cv2.imwrite("Thresh", thresh)
# cv2.waitKey(0)






from skimage.measure import compare_ssim
import cv2
import numpy as np


before = cv2.imread(args["first"])
after = cv2.imread(args["second"])


# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(before_gray, after_gray, full=True)
diff = 1 - (cv2.absdiff(before_gray, after_gray) / 255)

before_gray = before_gray.astype("int32")
after_gray = after_gray.astype("int32")

_subtract_ = before_gray - after_gray
_a_min_b_ = (_subtract_ <= 0.0).astype("uint8")

print(np.unique(_subtract_))
print(np.unique(_a_min_b_))

_subtract_ = after_gray - before_gray
_b_min_a_ = (_subtract_ <= 0.0).astype("uint8")

print("!!!")
print(f"{np.count_nonzero(_a_min_b_)} non zeros")
print(f"{np.count_nonzero(_b_min_a_)} non zeros")
print("!!!")

print(np.unique(_subtract_))
print(np.unique(_b_min_a_))


print(f"{np.count_nonzero(diff)} non zeros")
print(diff)

#cv2.imwrite('diff_Only.png',255*diff)



################################################################################
################################################################################
################################################################################
################################################################################
'''
Choose either 1st two or last two depending upon number of points needed 
'''

if args['all']:
	diff1 = _a_min_b_
	diff2 = _b_min_a_
else:
	diff1 = 1 - ((1 - diff) * _a_min_b_)
	diff2 = 1 - ((1 - diff) * _b_min_a_)


################################################################################
################################################################################
################################################################################
################################################################################











#cv2.bitwise_and(diff, diff, _a_min_b_)

print("\n\n")
print(diff)
print(f"{np.count_nonzero(diff)} non zeros")

#cv2.imwrite('diff___b_min_a_.png',255*diff)






print("Image similarity", score)

#print("Image similarity", np.unique(diff))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV

diff1 = (diff1 * 255).astype("uint8")
diff2 = (diff2 * 255).astype("uint8")


# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ

'''
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
'''

thresh1 = cv2.threshold(diff1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours1 = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

thresh2 = cv2.threshold(diff2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours1 = contours1[0] if len(contours1) == 2 else contours1[1]
contours2 = contours2[0] if len(contours2) == 2 else contours2[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours1:
    area = cv2.contourArea(c)
    if area > 5:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (12, 24, 251), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (12, 24, 251), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

for c in contours2:
    area = cv2.contourArea(c)
    if area > 5:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (255,0,255), -1)
        cv2.drawContours(filled_after, [c], 0, (255,0,255), -1)


label = args["first_label"]
bottom_center =  lambda img : (int(before.shape[0] / 2 - 50), int(before.shape[1] - 20))
cv2.putText(before, label, bottom_center(before), cv2.FONT_HERSHEY_PLAIN, 1.0, (12, 24, 251), 2);
cv2.putText(after, label, bottom_center(after), cv2.FONT_HERSHEY_PLAIN, 1.0, (12, 24, 251), 2);

label = args["second_label"]
bottom_center =  lambda img : (int(before.shape[0] / 2 + 50), int(before.shape[1] - 20))
cv2.putText(before, label, bottom_center(before), cv2.FONT_HERSHEY_PLAIN, 1.0, (36,255,12), 2);
cv2.putText(after, label, bottom_center(after), cv2.FONT_HERSHEY_PLAIN, 1.0, (36,255,12), 2);


cv2.imwrite('before.png', before)
cv2.imwrite('after.png', after)
cv2.imwrite('diff.png',diff)
cv2.imwrite('mask.png',mask)
cv2.imwrite('filled after.png',filled_after)


cv2.waitKey(0)