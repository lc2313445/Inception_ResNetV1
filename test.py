import tensorflow as tf # -*- coding: utf-8 -*-
import ops
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
ATEC_Model_Path='D:/xuexiziliao/Proj/atec/Graph_Save/Model/'
#ccc=tf.get_variable('ccc',initializer=tf.constant(2.0),trainable=False)


#with tf.Session() as sess:
 #   tf.global_variables_initializer().run()
    #sess.run(tf.assign(ccc,2.5))
    #saver.save(sess,ATEC_Model_Path+'model.ckpt',global_step=1)
    
    #saver.restore(sess,ATEC_Model_Path+'model.ckpt-1')
    #print(sess.run(ccc))
'''
label_batch_data=[[1,2],[3,4],[5,6]]
index=0
first_time=0
image_batch_data_select=[]
for label_index in label_batch_data:
    print('begin')
    print(label_index)
    if first_time==0:
        print('if')
        image_batch_data_select=[label_batch_data[index]]
        print(image_batch_data_select)
        first_time=1
    else:
        print('else')
        print(image_batch_data_select)
        print(label_batch_data[index])
        image_batch_data_select=np.append(image_batch_data_select,[label_batch_data[index]],0)
    
    index+=1
print(image_batch_data_select)
'''

#image_for_test=Image.open('D:/xuexiziliao/Proj\CNN_V1.0/V1_0/IMG/1.jpg')
#print(image_for_test.size)
image_for_test=mpimg.imread('D:/xuexiziliao/Proj\CNN_V1.0/V1_0/IMG/1.jpg')
print(image_for_test.shape)
a=image_for_test
m=[1,2,4]
plt.figure()
implot=plt.imshow(a)
for p in range(10):
    plt.text(p*100,100,'asd{}'.format(m))
plt.gca().add_patch(plt.Rectangle((500,1250),1000,800,fill=False,edgecolor='r', linewidth=0.5))
plt.gca().add_patch(plt.Rectangle((1500,450),1000,1450,fill=False,edgecolor='b', linewidth=0.5))
plt.gca().add_patch(plt.Rectangle((2400,1250),1000,850,fill=False,edgecolor='g', linewidth=0.5))
plt.scatter([100,200,300,400,500],[100,200,300,400,500],c='r',marker ='.')



'''
z=[]
for i in range(12):
    for j in range(12):
        x=a[i*252:(i+1)*252,j*336:(j+1)*336,:]
        print(np.shape(x))
        x=Image.fromarray(x)
        x=x.resize((160,120))
        x=np.asarray(x)
        z.append(x)
        
        plt.figure()
        plt.imshow(z[(i+1)*(j+1)-1])
z=np.array(z)
print(np.shape(z[0]))
print(np.shape(z))
'''
'''
reImg=[]

for i in range(np.shape(z)[0]):
    np.append(reImg,tf.image.resize_images(z[i],[126,168],3))
with tf.Session() as sess:
       Img=reImg.eval()
       plt.figure()
       plt.imshow(reImg[0])
 '''       
#a=np.array(image_for_test.getdata())
#plt.plot(a)
#print(a.shape)