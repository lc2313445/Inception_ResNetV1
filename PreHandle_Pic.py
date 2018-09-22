# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:48:10 2018

@author: Lee
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
Path="D:/xuexiziliao/Proj/VS PROJ/P1/RGB_IMG/Mouse/RGB 2018-4-28 10-0-47.jpg"
Path_Dir="D:/xuexiziliao/Proj/VS PROJ/P1/RGB_IMG/"
SAVE_PATH="D:/xuexiziliao/Proj/VS PROJ/P1/RGB_IMG/datasetBig.tfrecords"

def ASCII_To_Str(Input_List):
    Output_List=[]
    for j in Input_List:#input_Tensor=[['1','a',...],[],...]
        temp=''.join([chr(i) for i in j])
        Output_List.append(temp)
    return Output_List
'''
    Output_List=''
    for j in Input_List:#input_Tensor=[121, 221,...]
        temp=chr(j)
        Output_List=Output_List+temp
    return [Output_List]
'''

def Str_To_ASCII(Input_List):#'mouse'
    Output_List=[]
    for j in Input_List:
        temp=ord(j)
        Output_List.append(temp)
    return Output_List
        
    

def Picture2TFrecord():
    with tf.Session() as sess:        
        writer = tf.python_io.TFRecordWriter(SAVE_PATH)
        folders=os.listdir(Path_Dir)
        index_and_name=list(enumerate(folders))
        print(index_and_name)
        index=-1
        for index_include_files,folder_name in index_and_name:
            Path_Dir_Sub=Path_Dir+folder_name
            print(Path_Dir_Sub)
            if(os.path.isdir(Path_Dir_Sub)):
                print('ok')
                index=index+1
            else:
                continue
            for (root,sub_dir,filenames) in os.walk(Path_Dir_Sub):#自动判断是否为dir
                print("root {}, sub_dir {} ,filenames {}".format(root,sub_dir,filenames))
                for item in filenames:
                    Pic_path=os.path.join(root,item)
                    print(Pic_path)
                    image = tf.gfile.FastGFile(Pic_path, 'rb').read()
                    image = tf.image.decode_jpeg(image)
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
                    image=tf.image.resize_images(image,[120,160],3)
                    
                    image=image.eval()
                    print(np.shape(image))
                    image_raw=image.tostring()
                    folder_name=np.array(folder_name).tostring()
                    print('index: {}'.format(index))
                    example=tf.train.Example(features=tf.train.Features(feature={  
                            "label":
                                tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            "name":
                                tf.train.Feature(bytes_list=tf.train.BytesList(value=[folder_name])),
                            "image":
                                tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                                }))
                    writer.write(example.SerializeToString())
        writer.close()
        
        img=Image.open(Path)
        img=img.resize((64,48))
        print(np.shape(img))
        plt.subplot(223) 
        plt.imshow(img)
        plt.show()
def DecodeTFrecord(TFRecord_File,Batch_Size,Train_Or_Use='Train'):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([TFRecord_File])
    _,serialized_example=reader.read(filename_queue)#return file name and content
    
   
    features=tf.parse_single_example(serialized_example,
                                         features={
                                                 "label":
                                                     tf.FixedLenFeature([],tf.int64),
                                                 "name":
                                                     tf.FixedLenFeature([],tf.string),
                                                 "image":
                                                     tf.FixedLenFeature([],tf.string),
                                                     })
    label=tf.cast(features['label'],tf.int64)
    name=features['name']
    image=tf.decode_raw(features['image'],tf.float32)#figure must be float32
    image=tf.reshape(image,[57600,])#9216#230400
    
    
    input_queue = tf.train.slice_input_producer([[label],[name],[image]],shuffle=False)
    
    if(Train_Or_Use=='Use'):
        label_batch,name_batch,image_batch=tf.train.batch(input_queue,
                                                           batch_size=Batch_Size, 
                                                           num_threads=1,
                                                           capacity=3000,
                                                           allow_smaller_final_batch=True)
    else:
        label_batch,name_batch,image_batch=tf.train.shuffle_batch(input_queue,  
                                                                      batch_size=Batch_Size,
                                                                      num_threads=16,
                                                                      capacity=3000,
                                                                      min_after_dequeue=5)
        
    name_batch=tf.decode_raw(name_batch,tf.int32)
    
    return label_batch,name_batch,image_batch


def Decode2(TFRecord_File,Batch_Size,Train_Or_Use='Train'):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([TFRecord_File])
    _,serialized_example=reader.read(filename_queue)#return file name and content
    record_iterator=tf.python_io.tf_record_iterator(path=TFRecord_File)
    with tf.Session() as sess:
        for string_record in record_iterator:
            example=tf.train.Example()
            example.ParseFromString(string_record)
            name=(example.features.feature['name'])
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print(name)
    
    

def test_read_and_decode(Batch_Size): 
    label_batch,name_batch,image_batch = DecodeTFrecord(SAVE_PATH,Batch_Size,Train_Or_Use='Train')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()#线程协调器
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)#启动线程
        try:
            for i in range(5000):
                label_batch_data,name_batch_data,image_batch_data=sess.run([label_batch,name_batch,image_batch])
                image_batch_data=image_batch_data#float to uint8
                 
                print(label_batch_data)
                name_batch_data=ASCII_To_Str(name_batch_data)
                print(name_batch_data)
                
                print(np.shape(image_batch_data))
                #print(image_batch_data)
                image_batch_data=np.reshape(image_batch_data,(Batch_Size,240,320,3))
                print(np.shape(image_batch_data))
                print(i)
                '''
                for i in range(np.shape(image_batch_data)[0]):
                    image_batch_data[i]=sess.run(tf.image.per_image_standardization(image_batch_data[i]))
                #print(image_batch_data)
                '''
                plt.subplot(221)
                plt.imshow(np.abs(image_batch_data[0]))
                plt.subplot(222)
                plt.imshow(np.abs(image_batch_data[1]))
                plt.subplot(223)
                plt.imshow(np.abs(image_batch_data[2]))
                plt.show()
            
        except tf.errors.OutOfRangeError:
            print("done epcho.......................")
        finally:
            coord.request_stop()
        coord.join(threads)#回收线程


if __name__=='__main__':
    Picture2TFrecord()
    #Decode2(SAVE_PATH,1)
    #Picture2TFrecord()
    #test_read_and_decode(60)