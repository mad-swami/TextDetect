import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile

# read image 
image_path = '/home/mdetezanospinto/TextDetect/data/test4.png'

img = cv2.imread(image_path)

original_img = img.copy()
# preprocessing 

# normalizing
norm_img = np.zeros((img.shape[0], img.shape[1]))
img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

# skew correction

def deskew(image):
    co_ords = np.column_stack(np.where(image > 0))
    
    angle = cv2.minAreaRect(co_ords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

# img = deskew(img)

# Image Scaling

def set_image_dpi(file_path):
    im = Image.open(file_path)

    length_x, width_y = im.size

    factor = min(1, float(1024.0 / length_x))

    size = int(factor * length_x), int(factor * width_y)

    im_resized = im.resize(size, Image.Resampling.LANCZOS)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix = ".png")
    
    temp_filename = temp_file.name

    im_resized.save(temp_filename, dpi=(300, 300))

    return temp_filename
image_path = set_image_dpi(image_path)

# remove noise
def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
# img = remove_noise(img)

# grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = get_grayscale(img)

# threshold
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# img = thresholding(img)
# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# detect text on image
text_ = reader.readtext(img)

threshold = 0.30
# draw bbox and text
for t in text_:
    print(t)

    bbox, text, score = t

    # convert all bbox nums to ints
    bbox = [(int(x), int(y)) for x, y in bbox]

    if score > threshold:
        # if it is slanted (rotated box) use polylines
        cv2.polylines(img, [np.array(bbox)], isClosed=True, color = (0, 255, 0), thickness = 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
