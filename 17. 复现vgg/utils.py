import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

plt.rcParams["figure.figsize"] = (10, 8)

def load_image(image_path):
    img = Image.open(image_path)
    img_resize = img.resize((224, 224))
    img_ready = np.array(img_resize).reshape((1, 224, 224, 3))
    return img_ready / 255.0