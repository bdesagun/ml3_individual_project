#%%
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

import PIL
from PIL import Image
from itertools import groupby

from keras.preprocessing.image import img_to_array
import PIL.ImageOps


from bin.line_extractor import *


# defaul variables
Name = ['1', '+', '.', '==', '3', '2', '6', '/', '9', '*', '5', '0', '7', '-', '8', '4']
green = (114, 245, 66)
red = (252, 49, 45)
font = cv2.FONT_HERSHEY_SIMPLEX
correct_messages = ['good job', 'excellent', 'amazing', 'wonderful', 'great','correct','great']



#%%
model = keras.models.load_model("models/attempt2.h5")
#%%
def prediction(elements_array):
    # Give 2 a higher custom weight since it is always predicted as 7
    custom_weights = np.ones(16)
    custom_weights[[3,5,10]]= [9.7,19.8,2.5]

    p1 = [Image.fromarray(np.uint8(np.reshape(o,(40,40)) * 255),'L').convert('RGB') for o in elements_array]
    p2 = [np.expand_dims(np.array(img_to_array(PIL.ImageOps.invert(i))/255), axis=0) for i in p1]
    prediction= [Name[np.argmax(model.predict(p) * custom_weights)] for p in p2]
    # print([model.predict(p) for p in p2])
    return prediction

#%%


def check_paper(image_path):
    img = cv2.imread(image_path)
    clone = img.copy()

    # image preprocessing
    thresh = 120
    im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    img = im_bw

    cleaned_orig,y1s,y2s,rec_out = extract_line(img, show=False)
    backtorgb = 255 - cv2.cvtColor(cleaned_orig,cv2.COLOR_GRAY2RGB)


    # loop for every line
    # item_idx = 5
    # i  = rec_out[item_idx] 
    items = len(rec_out)
    score = 0
    for i in rec_out:
        img_interest = backtorgb[i[0][1]:i[1][1],i[0][0]:i[1][0]]
        # plt.imshow(img_interest)
        # plt.show()    

        image = Image.fromarray(img_interest).convert('RGB').convert("L")
        new_image_arr = np.array(image)
        reverse_img = 255 - new_image_arr
        ooooo = reverse_img[reverse_img.sum(axis=1).nonzero()]
        padding = np.zeros(shape=(5,reverse_img.shape[1]))
        image = np.concatenate((padding,ooooo,padding))

        image_i = PIL.ImageOps.invert(Image.fromarray(image).convert('RGB').convert('L'))

        # set height to 40px
        image = image_i
        w = image.size[0]
        h = image.size[1]
        r = w / h # aspect ratio
        new_w = int(r * 40)
        new_h = 40
        new_image = image.resize((new_w, new_h))

        # convert to grey scale
        new_image_arr = np.array(new_image)
        new_inv_image_arr = 255 - new_image_arr
        final_image_arr = new_inv_image_arr / 255.0
        m = final_image_arr.any(0)


        # Split the elements and set correct dimentions
        out = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]
        num_of_elements = len(out)
        elements_list = []
        for x in range(0, num_of_elements):
            img = out[x]
            
            #adding 0 value columns as fillers
            width = img.shape[1]
            filler = (final_image_arr.shape[0] - width) / 2
            
            if filler.is_integer() == False:    #odd number of filler columns
                filler_l = int(filler)
                filler_r = int(filler) + 1
            else:                               #even number of filler columns
                filler_l = int(filler)
                filler_r = int(filler)
            
            arr_l = np.zeros((final_image_arr.shape[0], filler_l)) #left fillers
            arr_r = np.zeros((final_image_arr.shape[0], filler_r)) #right fillers
            
            #concatinating the left and right fillers
            help_ = np.concatenate((arr_l, img), axis= 1)
            element_arr = np.concatenate((help_, arr_r), axis= 1)
            
            element_arr.resize(40, 40, 1) #resize array 2d to 3d
            #storing all elements in a list
            elements_list.append(element_arr)
        elements_array = np.array(elements_list)
        elements_array = elements_array.reshape(-1, 40, 40, 1)

        # Create the prediction
        precdiction_list = prediction(elements_array)

        # Python script
        eq = ''.join(precdiction_list)
        # print(eq)


        # cv2.rectangle(clone,rec_out[2][0],rec_out[2][1],green,2)
        # if eval(eq):
        #     cv2.putText(clone, np.random.choice(correct_messages), (i[0][0], i[1][1]-60), font, .7, green, 1, cv2.LINE_AA)
        # else:
        #     cv2.putText(clone, 'try again', (i[0][0], i[1][1]-60), font, .7, red, 1, cv2.LINE_AA)


        if eval(eq):
            # cv2.putText(clone, eq.replace("==", "="), (i[0][0], i[1][1]), font, .7, green, 1, cv2.LINE_AA)
            cv2.putText(clone, np.random.choice(correct_messages), (i[0][0], i[1][1]), font, .7, green, 1, cv2.LINE_AA)
            # cv2.rectangle(clone,i[0],i[1],green,1)
            score += 1
        else:
            # cv2.putText(clone, eq.replace("==", "="), (i[0][0], i[1][1]), font, .7, red, 1, cv2.LINE_AA)
            cv2.putText(clone, 'try again', (i[0][0], i[1][1]), font, .7, red, 1, cv2.LINE_AA)
            # cv2.rectangle(clone,i[0],i[1],red,1)



    # plt.figure(figsize=(12,12))
    # plt.axis('off')
    # plt.imshow(clone)
    plt.imsave('static/checked/{x}' .format(x=image_path.split('/')[-1]),clone)
    return items, score
# %%

# 2, 9 optional --7


# %%

