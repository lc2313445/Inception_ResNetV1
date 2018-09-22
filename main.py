import tensorflow as tf
import ops
import PreHandle_Pic
import Network
import numpy as np
import os
import psutil
import csv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Clique_Graph_Path='D:/xuexiziliao/Proj/CNN_V1.0/Graph_Save/Graph/'
Clique_Model_Path='D:/xuexiziliao/Proj/CNN_V1.0/Graph_Save/Model/'
Train_Data_TF="D:/xuexiziliao/Proj/VS PROJ/P1/RGB_IMG/datasetBig.tfrecords"
BATCH_SIZE=20
TRAIN_STEP=1000000000
Train_Or_Use='Train' #Train KeepTrain Use

    

def train():
    if Train_Or_Use=='Train' or Train_Or_Use=='KeepTrain':
        is_train=True
        label_batch,name_batch,image_batch=PreHandle_Pic.DecodeTFrecord(Train_Data_TF,(int)(BATCH_SIZE),Train_Or_Use='Train')
        label_batchV,name_batchV,image_batchV=PreHandle_Pic.DecodeTFrecord(Train_Data_TF,(int)(BATCH_SIZE),Train_Or_Use='Train')
    else:
        is_train=False
        #ID_batchT,label_batchT,date_batchT,features_data_batchT=PreHandle.read_and_decode(Train_Model_Test,BATCH_SIZE,Train_Or_Use=Train_Or_Use)
        #ID_batchR,date_batchR,features_data_batchR=PreHandle.read_and_decode(Real_Test_Data_TF,BATCH_SIZE,Train_Or_Use=Train_Or_Use)
        
    global_step=tf.Variable(0,name='global_step',trainable=False)
    discriminator=Network.Network()
    with tf.variable_scope('input_layer'): 
        x=tf.placeholder(tf.float32,[None,120,160,3],name='x-input')
        _y=tf.placeholder(tf.float32,[None,3],name='y-input')
        keep_prob = tf.placeholder(tf.float32)
        x_image=x
        #x_image=tf.reshape(x,[-1,48,64,3])
    with tf.variable_scope('Discriminator') as Disc:
        D=tf.clip_by_value(discriminator.NetGraph2(x_image,is_train=is_train,keep_prob=keep_prob,Res_Rate=1),-1e21,1e21)
    with tf.name_scope('LossAndAccuracy'):
        d_correct_prediction=tf.equal(tf.argmax(_y,1),tf.argmax(D,1))
        d_accuracy=tf.reduce_mean(tf.cast(d_correct_prediction,tf.float32),name='d_accuracy')
        #d_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D,labels=_y))
        d_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D,labels=_y))
        tf.add_to_collection('losses',d_loss)
        d_loss=tf.add_n(tf.get_collection('losses'))
    t_vars=tf.trainable_variables()
    d_vars=[var for var in t_vars if 'd_' in var.name]
    print(d_vars)
    learn_rate=tf.train.exponential_decay(0.00001,global_step,int(700000/BATCH_SIZE),0.9)
    with tf.variable_scope('Train',reuse=False):
        d_optim=tf.train.AdamOptimizer(learn_rate,beta1=0.9,beta2=0.9).minimize(d_loss,var_list=d_vars,global_step=global_step)
    with tf.name_scope('Summary'):
        d_sum=tf.summary.histogram('d',D)      
        n=[]
        for i in tf.get_collection('weight'):
            m=tf.reduce_sum(i)
            n=n+[m]
        weight_sum=tf.summary.histogram('weight',n)
        n=[]
        for i in tf.get_collection('bias'):
            m=tf.reduce_sum(i)
            n=n+[m]      
        bias_sum=tf.summary.histogram('bias',n)
        
         
        i=tf.get_collection('Inception_B')#[1,5,12,16,48]
        n=tf.reduce_mean(i,[4])
        n=tf.reshape(n,[BATCH_SIZE,tf.shape(n)[2],tf.shape(n)[3],1])
        conv_sum=tf.summary.image('conv_img',n,max_outputs=6)
        '''
        i=tf.get_collection('gn')
        gn_sum=tf.summary.image('gn',i[0],max_outputs=6)
        '''
        raw_image=tf.summary.image('raw_img',x_image,max_outputs=6)
        
        d_loss_sum=tf.summary.scalar('d_loss',d_loss)
        d_accuracy_sum=tf.summary.scalar('d_accuracy',d_accuracy)
        d_sum=tf.summary.merge([d_sum,d_loss_sum,d_accuracy_sum,weight_sum,bias_sum,raw_image,conv_sum])#conv_sum
        
        d_accuracy_sum_V1=tf.summary.scalar('d_accuracy_V1',d_accuracy)
        
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth=True
    saver=tf.train.Saver()
    
    
    if Train_Or_Use=='Train' or Train_Or_Use=='KeepTrain':
        with tf.Session(config=config) as sess:
            init=tf.global_variables_initializer()
            sess.run(init)
            writer=tf.summary.FileWriter(Clique_Graph_Path,sess.graph)
            print("RAM USE: %f"%(psutil.Process(os.getpid()).memory_info().rss/(1024*1024)))
            
            if(Train_Or_Use=="KeepTrain"):
                saver.restore(sess,Clique_Model_Path+'model.ckpt-11001')
                #Shadow_Accu0=tf.get_default_graph().get_tensor_by_name('Shadow_Accu0:0')
                #Shadow_Accu1=tf.get_default_graph().get_tensor_by_name('Shadow_Accu1:0')
             
            #read data''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            coord=tf.train.Coordinator()#线程协调器
            threads = tf.train.start_queue_runners(coord=coord,sess=sess)#启动线程
            try: 
                
                global_s=sess.run(global_step)
                for i in range(global_s,TRAIN_STEP):  
                    Should_Save=0
                    #while not coord.should_stop():
                    
                    label_batch_data,name_batch_data,image_batch_data=sess.run(
                                [label_batch,name_batch,image_batch])
                    label_batch_data=ops.one_hot(label_batch_data,3)
                    name_batch_data=np.array(PreHandle_Pic.ASCII_To_Str(name_batch_data))
                    image_batch_data=np.reshape(image_batch_data,[BATCH_SIZE,120,160,3]) 
                  
                    
                    global_s=sess.run(global_step)
                    print(i)
                    xs=image_batch_data
                    ys=label_batch_data

                    print('train...')
                    _,summary_str_d,Accu=sess.run([d_optim,d_sum,d_accuracy],feed_dict={x:xs,_y:ys,keep_prob:0.8})
                    print('d_accuracy in 50%% 0 and 1: %f'%Accu)
                    writer.add_summary(summary_str_d,global_s)
                     
                    
                    print('steps: %d global step: %d' %(i,global_s))
                    print('learn_rate: %f'%(sess.run(learn_rate)))
                    print('ys: {}'.format(ys))
                   #print(sess.run(D,feed_dict={x:xs,_y:ys,keep_prob:0.8}))
                    
                    
                    if(i%20==0):
                        run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata=tf.RunMetadata()
                        label_batch_dataV,name_batch_dataV,image_batch_dataV=sess.run(
                            [label_batchV,name_batchV,image_batchV])
                        name_batch_dataV=np.array(PreHandle_Pic.ASCII_To_Str(name_batch_dataV))
                        label_batch_dataV=ops.one_hot(label_batch_dataV,3)
                        image_batch_dataV=np.reshape(image_batch_dataV,[BATCH_SIZE,120,160,3])  
                      
                        D_loss=d_loss.eval({x:image_batch_dataV,_y:label_batch_dataV,keep_prob:1.0})
                        print('Train Step %d, d_loss: %.8f'%(i,D_loss))
                        for n,m,o in [sess.run([D,d_accuracy,d_accuracy_sum_V1],feed_dict={x:image_batch_dataV,_y:label_batch_dataV,keep_prob:1.0},
                                           options=run_options,run_metadata=run_metadata)]:
                            print('D:.......................{}'.format(n))
                            print('d_accuracy............................: %f'%m)
                            writer.add_summary(o,global_s)
                            
                        print('Train Step %d, d_loss: %.8f'%(i,D_loss))
                        
                            
                    if(global_s%200==0 and global_s//200>0 and Should_Save==0):
                        print('saving......')
                        saver.save(sess,Clique_Model_Path+'model.ckpt',global_step=global_step)
               
            except tf.errors.OutOfRangeError:
                print("done epcho.......................")
            finally:
                coord.request_stop()
            coord.join(threads)#回收线程
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    elif Train_Or_Use=='Use':
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess,Clique_Model_Path+'model.ckpt-11001')
            graph=tf.get_default_graph()
            for i in tf.global_variables():
                print(i)
            coord=tf.train.Coordinator()#线程协调器
            threads = tf.train.start_queue_runners(coord=coord,sess=sess)#启动线程
            demo=open('D:/xuexiziliao/Proj/atec/CSV_FILE/demo_for_test.csv','w',newline='')
            csv_writer=csv.writer(demo,dialect='excel')
            ID_batch_dataT_0=0
            finish_flag=0
            try: 
                csv_writer.writerow(['id','score'])
                for r in range(50):        
                    ID_batch_dataT,label_batch_dataT,date_batch_dataT,features_data_batch_dataT=sess.run(
                                    [ID_batchT,label_batchT,date_batchT,features_data_batchT])
                    ID_batch_dataT=np.array(PreHandle.ASCII_To_Str(ID_batch_dataT))
                    label_batch_dataT=ops.one_hot(label_batch_dataT,2)
                    
                    if(r==0):
                        ID_batch_dataT_0=ID_batch_dataT[0]
                        print(ID_batch_dataT_0)
                    elif(ID_batch_dataT_0 in ID_batch_dataT):
                        print('finish.....')
                        print(ID_batch_dataT_0)
                        finish_index=np.where(ID_batch_dataT_0==ID_batch_dataT)[0][0]
                        print(finish_index)
                        finish_flag=1
                        
                    
                    D_loss=d_loss.eval({x:features_data_batch_dataT,_y:label_batch_dataT})
                    print('Train Step %d, d_loss: %.8f'%(sess.run(global_step),D_loss))
                    for n,m in [sess.run([D,d_accuracy],feed_dict={x:features_data_batch_dataT,_y:label_batch_dataT})]:                   
                        print('D:.......................{}'.format(n))
                        print('Probility:.........{}'.format(ops.SoftMax(n)))
                        print('d_accuracy_ALL1............................: %f'%m)
                    Soft_Max=ops.SoftMax(n)
                    #Soft_Max=ops.select97(Soft_Max)
                    print('softMax:...............{}'.format(Soft_Max))
                    if(finish_flag==0):
                        length=len(n)
                    elif(finish_flag==1):
                        length=finish_index
                    for i in range(length):  
                        #csv_writer.writerow([ID_batch_dataT[i],Soft_Max[i][1]])
                        csv_writer.writerow([ID_batch_dataT[i],Soft_Max[i][1]])
                    print(r)
                    if(finish_flag==1):
                        break
                    
            except tf.errors.OutOfRangeError:
                print("done epcho.......................")
            finally:
                demo.close()
                coord.request_stop()
            coord.join(threads)#回收线程
                    
            
         
    
            
    elif Train_Or_Use=='Use_for_real':
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess,Clique_Model_Path+'model.ckpt-11001')
            graph=tf.get_default_graph()
            for i in tf.global_variables():
                print(i)
            
            
            image_for_test=mpimg.imread('D:/xuexiziliao/Proj\CNN_V1.0/V1_0/IMG/1.jpg')#numpy
            print(image_for_test.shape)
            '''
            image_for_test_Reshape=[]
            for i in range(12):
                for j in range(12):
                    temp=image_for_test[i*252:(i+1)*252,j*336:(j+1)*336,:]
                    temp=Image.fromarray(temp)#to Image
                    temp=temp.resize((160,120))
                    temp=np.asarray(temp)
                    image_for_test_Reshape.append(temp)
            image_for_test_Reshape=np.array(image_for_test_Reshape)
            '''
            image_for_test_Reshape=[]
            temp=image_for_test[1250:2050,500:1500,:]
            temp=Image.fromarray(temp)
            temp=temp.resize((160,120))
            temp=np.asarray(temp)
            plt.figure()
            plt.imshow(temp)
            image_for_test_Reshape.append(temp)
            
            temp=image_for_test[450:1900,1500:2500,:]
            temp=Image.fromarray(temp)
            temp=temp.resize((160,120))
            temp=np.asarray(temp)
            plt.figure()
            plt.imshow(temp)
            image_for_test_Reshape.append(temp)
            
            temp=image_for_test[1250:2100,2400:3400,:]
            temp=Image.fromarray(temp)
            temp=temp.resize((160,120))
            temp=np.asarray(temp)
            plt.figure()
            plt.imshow(temp)
            image_for_test_Reshape.append(temp)
            
            coord=tf.train.Coordinator()#线程协调器
            threads = tf.train.start_queue_runners(coord=coord,sess=sess)#启动线程
            '''
            demo=open('D:/xuexiziliao/Proj/atec/CSV_FILE/demo.csv','w',newline='')
            demo1=open('D:/xuexiziliao/Proj/atec/CSV_FILE/demo1.csv','w',newline='')
            demo2=open('D:/xuexiziliao/Proj/atec/CSV_FILE/demo2.csv','w',newline='')
            csv_writer=csv.writer(demo,dialect='excel')
            csv_writer1=csv.writer(demo1,dialect='excel')
            csv_writer2=csv.writer(demo2,dialect='excel')
            '''
            ID_batch_dataT_0=0
            finish_flag=0
            try: 
                    
                for n in [sess.run(D,feed_dict={x:image_for_test_Reshape,keep_prob:1})]:                   
                    print('D:.......................{}'.format(n))
                    #print('Probility:.........{}'.format(ops.SoftMax(n)))
                print(np.shape(n))
                Soft_Max=ops.SoftMax(n)
                Max=sess.run(tf.argmax(n,1))
                print('softMax:...............{}'.format(Soft_Max))
                print('Max:...............{}'.format(Max))
                
                plt.figure()
                plt.imshow(image_for_test)  
                '''
                for i in range(12):
                    for j in range(12): 
                        plt.text(i*336,j*252,'{}'.format(n[i*12+j]))
                '''
                plt.show()
            except tf.errors.OutOfRangeError:
                print("done epcho.......................")
            finally:
                coord.request_stop()
            coord.join(threads)#回收线程
            
            
            
        
if __name__=='__main__':
    train()
    '''
    if(Train_Or_Use=='Train' or Train_Or_Use=='KeepTrain'):
        train()
    if(Train_Or_Use=='Use'):
        saver=tf.train.import_meta_graph(ATEC_Model_Path+'model.ckpt-201.meta')#load graph
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess,ATEC_Model_Path+'model.ckpt-201')
            graph=tf.get_default_graph()
            for i in tf.global_variables():
                print(i)
            x=graph.get_tensor_by_name('input_layer/x-input:0')
            y_=graph.get_tensor_by_name('input_layer/y-input:0')
            d_accuracy=graph.get_tensor_by_name('LossAndAccuracy/d_accuracy:0')
            #xt,yt,kbt=feed_dict(False,1.0)
            #print("Accuracy d: %f g: %f"%(sess.run([d_accuracy,g_accuracy],feed_dict={x:xt,y_:yt})))
    '''
            

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    