from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

generator=ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

data_path='dataset - Copy'
target=os.listdir(data_path)
c=0
for i in target:
    path=os.path.join(data_path, i)
    catg=os.listdir(path)
    for im in catg:
        img=load_img(path+'\\'+im, target_size=(125, 125), color_mode='grayscale')
        _, img_format=os.path.splitext(im)
        img_format=img_format.replace('.', '')
        img=img_to_array(img)
        img=img.reshape(1, 125, 125, 1)

        count=0
        save_path = 'dataset'+'\\'+i
        for batch in generator.flow(img, batch_size=5, save_to_dir=save_path, save_format=img_format):
            count+=1
            if count==10:
                break

