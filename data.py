import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array

data_path='dataset'
targets=['with mask', 'without mask']
target_dict= {targets[i]:i for i in range(len(targets))}

x=[]
y=[]
for target in targets:
    path=os.path.join(data_path, target)
    lib=os.listdir(path)
    for image in lib:
        img=load_img(path+'\\'+image, target_size=(125, 125), color_mode='grayscale')
        img=img_to_array(img).astype('float32')/255.0
        img=img.reshape(125, 125)
        x.append(img)
        y.append(target_dict[target])

np.save('training_data/data.npy', x)
np.save('training_data/target.npy', y)