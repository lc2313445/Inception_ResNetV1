import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import math


def get_weights(shape,lamb):
    weights=tf.get_variable('weights',
                                    shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamb)(weights))
    tf.add_to_collection('weight',weights)
    return weights
def get_bias(shape):
    biases=tf.get_variable('biases',[shape],initializer=tf.constant_initializer(0.0))
    tf.add_to_collection('bias',biases)
    return biases
def batch_norm_layer(value,is_train=True,name='batch_norm'):#value [N,h,w,C] [N,C]
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value,decay=0.9,epsilon=1e-5,scale=True,is_training=is_train,
                              updates_collections=None,scope=scope)
        else:
            return batch_norm(value,decay=0.9,epsilon=1e-5,scale=True,is_training=is_train,
                              updates_collections=None,scope=scope)
def linear_layer(value,output_dim,name='linear_connected'):
    with tf.variable_scope(name):
        try:
            weights=get_weights([value.shape[1],output_dim],0.1**20)
            biases=get_bias(output_dim)
           # biases=tf.get_variable('biases',
           #                      [output_dim],initializer=tf.constant_initializer(0.0))
        except ValueError:
            print('linear_layer Value Error,name space: %s'%name)
            tf.get_variable_scope().reuse_variables()
            weights=get_weights([value.shape[1],output_dim],0.1**20)
            biases=get_bias(output_dim)
        return tf.matmul(value,weights)+biases

def conv2d(value,output_dim,kernal_h=5,kernal_w=5,strides=[1,1,1,1],name='conv2d'):
    with tf.variable_scope(name):
        try:
            weights=get_weights([kernal_h,kernal_w,int(value.shape[-1]),output_dim],0.1**20)
            biases=get_bias(output_dim)
        except ValueError:
            print('conv_layer Value Error,name space: %s'%name)
            tf.get_variable_scope().reuse_variables()
            weights=get_weights([kernal_h,kernal_w,int(value.shape[-1]),output_dim],0.1**20)
            biases=get_bias(output_dim)
        conv=tf.nn.conv2d(value,weights,strides=strides,padding='SAME')
        conv=tf.nn.bias_add(conv,biases)
        return conv
    
def average_pool(input_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='Avg_Pool'):
    return tf.nn.avg_pool(input_tensor,ksize=ksize,strides=strides,padding=padding,name=name)
def max_pool(input_tensor,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='Max_Pool'):
    return tf.nn.max_pool(input_tensor,ksize=ksize,strides=strides,padding=padding,name=name)


def Conv_Layer0(var_input,var_output=16,name='d_Conv_Layer0',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Level1'):
            level1=conv2d(var_input,64,7,7,[1,1,1,1],name='Conv2d_l1_7x7')#(48x64)/2
            #bn=lrelu(ops.batch_norm_layer(level1,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level1)
        with tf.variable_scope('Level2'):
            level2=conv2d(bn,var_output,3,3,[1,1,1,1],name='Conv2d_l2_3x3')
            #bn=lrelu(ops.batch_norm_layer(level2,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level2)
            #bn=average_pool(bn,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',name='Avg_Pool')
        return bn

def Conv_Layer0_1(var_input,var_output=16,name='d_Conv_Layer0',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Level1'):
            level1=conv2d(var_input,48,5,5,[1,1,1,1],name='Conv2d_l1_5x5')#(48x64)/2
            #bn=lrelu(ops.batch_norm_layer(level1,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level1)
        with tf.variable_scope('Level2'):
            level2=conv2d(bn,var_output,3,3,[1,1,1,1],name='Conv2d_l2_3x3')
            #bn=lrelu(ops.batch_norm_layer(level2,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level2)
            #bn=average_pool(bn,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',name='Avg_Pool')
        return bn



def Conv_Layer1(var_input,var_output=16,name='d_Conv_Layer1',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Level1'):
            level1=conv2d(var_input,64,7,7,[1,1,1,1],name='Conv2d_l1_7x7')#(48x64)/2
            #bn=lrelu(ops.batch_norm_layer(level1,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level1)
        with tf.variable_scope('Level2'):
            level2=conv2d(bn,var_output,3,3,[1,2,2,1],name='Conv2d_l2_3x3')
            #bn=lrelu(ops.batch_norm_layer(level2,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level2)
            #bn=average_pool(bn,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='Avg_Pool')
        return bn
    
def Conv_Layer2(var_input,var_output=16,name='d_Conv_Layer2',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Level1'):
            level1=conv2d(var_input,32,3,3,[1,1,1,1],name='Conv2d_l1_7x7')#(48x64)/2
            #bn=lrelu(ops.batch_norm_layer(level1,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level1)
        with tf.variable_scope('Level2'):
            level2=conv2d(bn,var_output,3,3,[1,2,2,1],name='Conv2d_l2_3x3')
            #bn=lrelu(ops.batch_norm_layer(level2,is_train=is_train,name='batch_norm_bn'))
            bn=lrelu(level2)
            #bn=max_pool(bn,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='Max_Pool')
        return bn
   
#inception_resnet v2 input
def input_Conv_Layer(var_input,var_in=32,name='d_input_Conv_Layer',is_train=True):
    with tf.variable_scope(name):
        Cell0=conv2d(var_input,var_in,3,3,[1,2,2,1],name='Conv2d_0_3x3')#[-1,60,80,32]
        Cell1=conv2d(Cell0,var_in,3,3,[1,1,1,1],name='Conv2d_1_3x3')#[-1,60,80,32]
        Cell2=conv2d(Cell1,var_in*2,3,3,[1,1,1,1],name='Conv2d_2_3x3')#[-1,60,80,64]
        MaxPool_Level=max_pool(Cell2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='Max_Pool_l1')#[-1,30,40,64]
        Cell3=conv2d(Cell2,var_in*3,3,3,[1,2,2,1],name='Conv2d_3_3x3')#[-1,30,40,96]
        concat_level=tf.concat([MaxPool_Level,Cell3],3)#[-1,30,40,64+96=160]
        #'l2'''''''''''''''''''''''
        Cell1_0=conv2d(concat_level,var_in*2,1,1,[1,1,1,1],name='Conv2d_1_0_1x1')#[-1,30,40,64]
        Cell1_1=conv2d(Cell1_0,var_in*3,3,3,[1,1,1,1],name='Conv2d_1_1_3x3')#[-1,30,40,96]
        
        Cell2_0=conv2d(concat_level,var_in*2,1,1,[1,1,1,1],name='Conv2d_2_0_1x1')#[-1,30,40,64]
        Cell2_1=conv2d(Cell2_0,var_in*2,7,1,[1,1,1,1],name='Conv2d_2_1_7x1')#[-1,30,40,64]
        Cell2_2=conv2d(Cell2_1,var_in*2,1,7,[1,1,1,1],name='Conv2d_2_2_1x7')#[-1,30,40,64]
        Cell2_3=conv2d(Cell2_2,var_in*3,3,3,[1,1,1,1],name='Conv2d_2_3_3x3')#[-1,30,40,96]
        concat_level=tf.concat([Cell1_1,Cell2_3],3)#[-1,60,80,96+96=192]
        #'l3'''''''''''''''''''''''
        Cell3_0=conv2d(concat_level,var_in*6,3,3,[1,1,1,1],name='Conv2d_3_0_3x3')#[-1,30,40,192]
        Cell3_1=max_pool(concat_level,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',name='Max_Pool_l3')##[-1,30,40,192]
        concat_level=tf.concat([Cell3_0,Cell3_1],3)#[-1,30,40,192+192=384]
        #Add Relu outside
        return concat_level

def Inception_ResNet_A(var_input,var_in=32,Res_Rate=0.1,name='Inception_ResNet_A1',is_train=True):
#Var_input should be after Relu, Figure 16
    with tf.variable_scope(name):
        Cell0_0=conv2d(var_input,var_in,1,1,[1,1,1,1],name='Conv2d_0_0_1x1')#[-1,30,40,32]
        
        Cell1_0=conv2d(var_input,var_in,1,1,[1,1,1,1],name='Conv2d_1_0_1x1')#[-1,30,40,32]
        Cell1_1=conv2d(Cell1_0,var_in,3,3,[1,1,1,1],name='Conv2d_1_1_3x3')#[-1,30,40,32]
        
        Cell2_0=conv2d(var_input,var_in,1,1,[1,1,1,1],name='Conv2d_2_0_1x1')#[-1,30,40,32]
        Cell2_1=conv2d(Cell2_0,3*var_in/2,3,3,[1,1,1,1],name='Conv2d_2_1_3x3')#[-1,30,40,48]
        Cell2_2=conv2d(Cell2_1,2*var_in,3,3,[1,1,1,1],name='Conv2d_2_2_3x3')#[-1,30,40,64]
        
        concat_level=tf.concat([Cell0_0,Cell1_1,Cell2_2],3)#[-1,30,40,32+32+64=128]
        Cell_Total=conv2d(concat_level,12*var_in,1,1,[1,1,1,1],name='Conv2d_Total_1x1')#[-1,30,40,384]
        
        Res_Add=tf.add_n([Res_Rate*var_input,Cell_Total])
        return Res_Add
    
    
def Inception_ResNet_B(var_input,var_in=32,Res_Rate=0.1,name='Inception_ResNet_B1',is_train=True):
#Var_input should be after Relu, Figure 17
    with tf.variable_scope(name):
        Cell0_0=conv2d(var_input,var_in*6,1,1,[1,1,1,1],name='Conv2d_0_0_1x1')#[-1,15,20,192]
        
        Cell1_0=conv2d(var_input,var_in*4,1,1,[1,1,1,1],name='Conv2d_1_0_1x1')#[-1,15,20,128]
        Cell1_1=conv2d(Cell1_0,var_in*5,1,7,[1,1,1,1],name='Conv2d_1_1_1x7')#[-1,15,20,224]
        Cell1_2=conv2d(Cell1_1,var_in*6,7,1,[1,1,1,1],name='Conv2d_1_2_7x1')#[-1,15,20,256]
        
        concat_level=tf.concat([Cell0_0,Cell1_2],3)#[-1,15,20,192+256=448]
        Cell_Total=conv2d(concat_level,var_in*36,1,1,[1,1,1,1],name='Conv2d_Total_1x1')#[-1,15,20,1152]
        
        Res_Add=tf.add_n([Res_Rate*var_input,Cell_Total])
        return Res_Add
    
def Inception_ResNet_C(var_input,var_in=32,Res_Rate=0.1,name='Inception_ResNet_C1',is_train=True):
#Var_input should be after Relu, Figure 19
    with tf.variable_scope(name):
        Cell0_0=conv2d(var_input,var_in*6,1,1,[1,1,1,1],name='Conv2d_0_0_1x1')#[-1,8,10,192]
        
        Cell1_0=conv2d(var_input,var_in*6,1,1,[1,1,1,1],name='Conv2d_1_0_1x1')#[-1,8,10,192]
        Cell1_1=conv2d(Cell1_0,var_in*7,1,3,[1,1,1,1],name='Conv2d_1_1_1x3')#[-1,8,10,224]
        Cell1_2=conv2d(Cell1_1,var_in*8,3,1,[1,1,1,1],name='Conv2d_1_2_3x1')#[-1,8,10,256]
        
        concat_level=tf.concat([Cell0_0,Cell1_2],3)#[-1,15,20,192+256=448]
        Cell_Total=conv2d(concat_level,var_in*67,1,1,[1,1,1,1],name='Conv2d_Total_1x1')#[-1,8,10,2144]
        
        Res_Add=tf.add_n([Res_Rate*var_input,Cell_Total])#[-1,8,10,2144]
        return Res_Add

def Selection(var_input,var_out=500,name='Selection_Layer',is_train=True):
    with tf.variable_scope(name):
        Reduction=conv2d(var_input,var_out,1,1,[1,1,1,1],name='Selection')#[-1,15,20,500]
    return Reduction
    
#redurction A Fig7
def Reduction_A(var_input,var_in=32,name='Reduction_A1',is_train=True):
    with tf.variable_scope(name):
        Cell0_0=max_pool(var_input,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='Max_Pool_0')##[-1,15,20,384]
        
        Cell1_0=conv2d(var_input,var_in*12,3,3,[1,2,2,1],name='Conv2d_1_0_3x3')#[-1,15,20,384]
        
        Cell2_0=conv2d(var_input,var_in*8,1,1,[1,1,1,1],name='Conv2d_2_0_1x1')#[-1,30,40,256]
        Cell2_1=conv2d(Cell2_0,var_in*8,3,3,[1,1,1,1],name='Conv2d_2_1_3x3')#[-1,30,40,256]
        Cell2_2=conv2d(Cell2_1,var_in*12,3,3,[1,2,2,1],name='Conv2d_2_2_3x3')#[-1,15,20,384]
        
        concat_level=tf.concat([Cell0_0,Cell1_0,Cell2_2],3)#[-1,30,40,384+384+256=1024]
        return concat_level
    
def Reduction_B(var_input,var_in=32,name='Reduction_B1',is_train=True):
    with tf.variable_scope(name):
        Cell0_0=max_pool(var_input,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='Max_Pool_0')##[-1,8,10,1154]
        
        Cell1_0=conv2d(var_input,var_in*8,1,1,[1,1,1,1],name='Conv2d_1_0_1x1')#[-1,15,20,256]
        Cell1_1=conv2d(Cell1_0,var_in*12,3,3,[1,2,2,1],name='Conv2d_1_1_1x1')#[-1,8,10,384]
        
        Cell2_0=conv2d(var_input,var_in*8,1,1,[1,1,1,1],name='Conv2d_2_0_1x1')#[-1,15,20,256]
        Cell2_1=conv2d(Cell2_0,var_in*9,3,3,[1,2,2,1],name='Conv2d_2_1_1x1')#[-1,8,10,288]
        
        Cell3_0=conv2d(var_input,var_in*8,1,1,[1,1,1,1],name='Conv2d_3_0_1x1')#[-1,15,20,256]
        Cell3_1=conv2d(Cell3_0,var_in*9,3,3,[1,1,1,1],name='Conv2d_3_1_1x1')#[-1,15,20,288]
        Cell3_2=conv2d(Cell3_1,var_in*10,3,3,[1,2,2,1],name='Conv2d_3_2_1x1')#[-1,8,10,320]
        
        concat_level=tf.concat([Cell0_0,Cell1_1,Cell2_1,Cell3_2],3)#[-1,8,10,1154+384+288+320=2144]
        return concat_level








def Linear_Layer_For_Incep_Res(var_input,name='Linear',is_train=True,keep_prob=1):
     with tf.variable_scope(name):
        linear1=linear_layer(var_input,1000,name='d_linear_layer1')#[-1,1000]
        linear1_prob = tf.nn.dropout(linear1, keep_prob)  
        
        #bn=ops.batch_norm_layer(linear1,is_train=is_train,name='batch_norm_bn1')
        #bn=group_norm_layer(linear1,50,1,0,name='d_bn_linear1')
        lr=lrelu(linear1_prob,name='d_lrelu_linear1')#[-1,1000]
        linear2=linear_layer(lr,500,name='d_linear_layer2')#[-1,500]
        #bn=bn=ops.batch_norm_layer(linear2,is_train=is_train,name='batch_norm_bn2')
        #bn=group_norm_layer(linear2,50,1,0,name='d_bn_linear2')
        lr=lrelu(linear2,name='d_lrelu_linear2')#[-1,500]    
        
        linear3=linear_layer(lr,100,name='d_linear_layer3')#[-1,3]   
        #bn=bn=ops.batch_norm_layer(linear3,is_train=is_train,name='batch_norm_bn3')
        lr=lrelu(linear3,name='d_lrelu_linear3')#[-1,2]
        
        linear4=linear_layer(lr,3,name='d_linear_layer4')#[-1,3]   
        #bn=bn=ops.batch_norm_layer(linear4,is_train=is_train,name='batch_norm_bn4')
        lr=lrelu(linear4,name='d_lrelu_linear4')#[-1,2]
        #SoftMax=5*tf.nn.softmax(lr)#0~5, sigmoid 0.5~0.9933 1/(1+exp(-5))
        return lr



        





def Individual_Cell(var_input,var_output=16,w=3,h=3,name='Individual_Cell',is_train=True):
    with tf.variable_scope(name):
        conv=conv2d(var_input,var_output,w,h,name=name)
        return lrelu(conv)

def Block(var_input,inout_node=16,name='d_block1',is_train=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        
        Cell_0=Individual_Cell(var_input,var_output=inout_node,w=1,h=1,name='Cell_0',is_train=is_train)
        
        Cell_1=Individual_Cell(Cell_0,var_output=inout_node,w=7,h=7,name='Cell_1',is_train=is_train)
        Cell_2=Individual_Cell(Cell_0,var_output=inout_node,w=3,h=3,name='Cell_2',is_train=is_train)
        Cell_3=Individual_Cell(Cell_0,var_output=inout_node,w=5,h=5,name='Cell_3',is_train=is_train)
        Cell_4=Individual_Cell(Cell_0,var_output=inout_node,w=3,h=3,name='Cell_4',is_train=is_train)
        
        '''
        c1 16 16
        c2 16 32
        c3 16 16
        c4 16 16
        
        '''
        print(tf.get_variable_scope().name)
       # tf.get_variable_scope().reuse_variables()
       #Should reuse
        '''Cell1 to others'''
        
        Cell_1_2=Individual_Cell(Cell_1,var_output=inout_node,w=3,h=3,name='Cell_2',is_train=is_train)
        Cell_1_3=Individual_Cell(Cell_1,var_output=inout_node,w=5,h=5,name='Cell_3',is_train=is_train)
        Cell_1_4=Individual_Cell(Cell_1,var_output=inout_node,w=3,h=3,name='Cell_4',is_train=is_train)
        '''Cell2 to others'''
        Cell_2_1=Individual_Cell(Cell_2,var_output=inout_node,w=7,h=7,name='Cell_1',is_train=is_train)
        Cell_2_3=Individual_Cell(Cell_2,var_output=inout_node,w=5,h=5,name='Cell_3',is_train=is_train)
        Cell_2_4=Individual_Cell(Cell_2,var_output=inout_node,w=3,h=3,name='Cell_4',is_train=is_train)
        '''Cell3 to others'''
        Cell_3_1=Individual_Cell(Cell_3,var_output=inout_node,w=7,h=7,name='Cell_1',is_train=is_train)
        Cell_3_2=Individual_Cell(Cell_3,var_output=inout_node,w=3,h=3,name='Cell_2',is_train=is_train)
        Cell_3_4=Individual_Cell(Cell_3,var_output=inout_node,w=3,h=3,name='Cell_4',is_train=is_train)
        '''Cell4 to others'''
        Cell_4_1=Individual_Cell(Cell_4,var_output=inout_node,w=7,h=7,name='Cell_1',is_train=is_train)
        Cell_4_2=Individual_Cell(Cell_4,var_output=inout_node,w=3,h=3,name='Cell_2',is_train=is_train)
        Cell_4_3=Individual_Cell(Cell_4,var_output=inout_node,w=5,h=5,name='Cell_3',is_train=is_train)
        '''
        Flow_back=tf.concat([Cell_1,Cell_2_1,Cell_3_1,Cell_4_1,
                             Cell_2,Cell_1_2,Cell_3_2,Cell_4_2,
                             Cell_3,Cell_1_3,Cell_2_3,Cell_4_3
                             Cell_4,Cell_1_4,Cell_2_4,Cell_3_4],3)#out 16*16
    '''
        Cell_All=tf.concat([Cell_1,Cell_2_1,Cell_3_1,Cell_4_1,
                                 Cell_2,Cell_1_2,Cell_3_2,Cell_4_2,
                                 Cell_3,Cell_1_3,Cell_2_3,Cell_4_3,
                                 Cell_4,Cell_1_4,Cell_2_4,Cell_3_4],3)#out 16*16
        
        #Cell_All_0=Individual_Cell(Flow_back,var_output=inout_node,w=1,h=1,name='Cell_0',is_train=is_train)
    return Cell_All

def Transition(var_input,w,h,var_output=16,name='Transition1',is_train=True):
     with tf.variable_scope(name):
         out=Individual_Cell(var_input,var_output=var_output,w=w,h=h,name=name,is_train=is_train)
     return out

def Cell_Incep1(var_input,name='Cell_Inception1',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            branch_0=conv2d(var_input,16,1,1,[1,1,1,1],name='Conv2d_0a_1x1')#[1,1,-1,16]
        with tf.variable_scope('Branch_1'):
            branch_1=conv2d(var_input,16,1,1,[1,1,1,1],name='Conv2d_0a_1x1')
            branch_1=lrelu(branch_1,name='Conv2d_0a_1x1_relu')
            branch_1=tf.concat([
                conv2d(branch_1,16,1,3,[1,1,1,1],'Conv2d_1a_1x3'),
                conv2d(branch_1,16,3,1,[1,1,1,1],'Conv2d_1b_3x1')],3)#[-1,32,32,32]
        with tf.variable_scope('Branch_2'):
            branch_2=conv2d(var_input,32,1,1,[1,1,1,1],'Conv2d_0a_1x1')
            branch_2=lrelu(branch_2,name='Conv2d_0a_1x1_relu')
            branch_2=conv2d(branch_2,16,3,3,[1,1,1,1],'Conv2d_1a_3x3')
            branch_2=lrelu(branch_2,name='Conv2d_1a_3x3_relu')
            branch_2=tf.concat([conv2d(branch_2,16,1,3,[1,1,1,1],'Conv2d_2a_1x3'),
                                conv2d(branch_2,16,3,1,[1,1,1,1],'Conv2d_2b_3x1')],3)#[-1,11,27,32]
        with tf.variable_scope('Branch_3'):
            branch_3=average_pool(var_input,ksize=[1,3,3,1],strides=[1,1,1,1],name='AvgPool_0a_3x3')
            branch_3=lrelu(branch_3,name='AvgPool_0a_3x3_relu')
            branch_3=conv2d(branch_3,16,1,1,[1,1,1,1],'Conv2d_1a_1x1')#[-1,11,27,16]
        Concat_Temp=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        #bn=ops.batch_norm_layer(Concat_Temp,is_train=is_train,name='batch_norm_bn')
        return lrelu(Concat_Temp)

def Cell_Incep2(var_input,name='Cell_Inception2',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            branch_0=tf.concat(
                    [conv2d(var_input,16,3,1,[1,1,1,1],name='Conv2d_0a_3x1'),
                    conv2d(var_input,16,1,3,[1,1,1,1],name='Conv2d_0b_1x3')],3)#[1,1,-1,32]
            branch_0=lrelu(branch_0,name='Conv2d_0b_1x3_3x1_relu')
            branch_0=tf.concat(
                    [conv2d(branch_0,16,1,3,[1,1,1,1],name='Conv2d_1a_1x3'),
                    conv2d(branch_0,16,3,1,[1,1,1,1],name='Conv2d_1b_3x1')],3)#[1,1,-1,32]
            branch_0=lrelu(branch_0,name='Conv2d_0b_1x3_3x1_2_relu')
            
            branch_0=tf.concat(
                    [conv2d(branch_0,16,3,1,[1,1,1,1],name='Conv2d_2a_3x1'),
                    conv2d(branch_0,16,1,3,[1,1,1,1],name='Conv2d_2b_1x3')],3)#[1,1,-1,32]
            
        with tf.variable_scope('Branch_1'):
            branch_1=conv2d(var_input,16,1,1,[1,1,1,1],name='Conv2d_0a_1x1')
            branch_1=lrelu(branch_1,name='Conv2d_0a_1x1_relu')
            branch_1=conv2d(branch_1,16,2,2,[1,1,1,1],name='Conv2d_1a_2x2')#[1,1,-1,16]
        with tf.variable_scope('Branch_2'):
            branch_2=conv2d(var_input,32,1,1,[1,1,1,1],'Conv2d_0a_1x1')
            branch_2=lrelu(branch_2,name='Conv2d_0a_1x1_relu')
            branch_2=conv2d(branch_2,16,3,3,[1,1,1,1],'Conv2d_1a_3x3')
            branch_2=lrelu(branch_2,name='Conv2d_1a_3x3_relu')
            branch_2=tf.concat([conv2d(branch_2,16,1,3,[1,1,1,1],'Conv2d_2a_1x3'),
                                conv2d(branch_2,16,3,1,[1,1,1,1],'Conv2d_2b_3x1')],3)#[1,1,-1,32]
        with tf.variable_scope('Branch_3'):
            branch_3=max_pool(var_input,ksize=[1,3,3,1],strides=[1,1,1,1],name='AvgPool_0a_3x3')
            branch_3=lrelu(branch_3,name='AvgPool_0a_3x3_relu')
            branch_3=conv2d(branch_3,16,1,1,[1,1,1,1],'Conv2d_1a_1x1')#[-1,-1,16]
        Concat_Temp=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        #bn=ops.batch_norm_layer(Concat_Temp,is_train=is_train,name='batch_norm_bn')
        return lrelu(Concat_Temp)#[96]


def Cell_Incep3(var_input,name='Cell_Inception3',is_train=True):
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            branch_0=tf.concat(
                    [conv2d(var_input,8,3,1,[1,1,1,1],name='Conv2d_0a_3x1'),
                    conv2d(var_input,8,1,3,[1,1,1,1],name='Conv2d_0b_1x3')],3)#[1,1,-1,16]
            branch_0=lrelu(branch_0,name='Conv2d_0b_1x3_3x1_relu')
            branch_0=tf.concat(
                    [conv2d(branch_0,8,1,3,[1,1,1,1],name='Conv2d_1a_1x3'),
                    conv2d(branch_0,8,3,1,[1,1,1,1],name='Conv2d_1b_3x1')],3)#[1,1,-1,16]
            branch_0=lrelu(branch_0,name='Conv2d_0b_1x3_3x1_2_relu')
            
            branch_0=tf.concat(
                    [conv2d(branch_0,16,3,1,[1,1,1,1],name='Conv2d_2a_3x1'),
                    conv2d(branch_0,16,1,3,[1,1,1,1],name='Conv2d_2b_1x3')],3)#[1,1,-1,16]
            
        with tf.variable_scope('Branch_1'):
            branch_1=conv2d(var_input,8,1,1,[1,1,1,1],name='Conv2d_0a_1x1')
            branch_1=lrelu(branch_1,name='Conv2d_0a_1x1_relu')
            branch_1=conv2d(branch_1,8,2,2,[1,1,1,1],name='Conv2d_1a_2x2')#[1,1,-1,8]
        with tf.variable_scope('Branch_2'):
            branch_2=conv2d(var_input,16,1,1,[1,1,1,1],'Conv2d_0a_1x1')
            branch_2=lrelu(branch_2,name='Conv2d_0a_1x1_relu')
            branch_2=conv2d(branch_2,8,3,3,[1,1,1,1],'Conv2d_1a_3x3')
            branch_2=lrelu(branch_2,name='Conv2d_1a_3x3_relu')
            branch_2=tf.concat([conv2d(branch_2,8,1,3,[1,1,1,1],'Conv2d_2a_1x3'),
                                conv2d(branch_2,8,3,1,[1,1,1,1],'Conv2d_2b_3x1')],3)#[1,1,-1,32]
        with tf.variable_scope('Branch_3'):
            branch_3=max_pool(var_input,ksize=[1,3,3,1],strides=[1,1,1,1],name='AvgPool_0a_3x3')
            branch_3=lrelu(branch_3,name='AvgPool_0a_3x3_relu')
            branch_3=conv2d(branch_3,8,2,2,[1,1,1,1],'Conv2d_1a_1x1')#[-1,-1,8]
        Concat_Temp=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        #bn=ops.batch_norm_layer(Concat_Temp,is_train=is_train,name='batch_norm_bn')
        return lrelu(Concat_Temp)#[48]






def SPP(var_input,name='SPP',is_train=True):
    var_shape=np.shape(var_input)
    print(var_shape)
        
    
    with tf.variable_scope(name):
        with tf.variable_scope('SPP4x4'):
            w=(var_shape[1]+4-1)//4  #4*4 maxpool with round up
            h=(var_shape[2]+4-1)//4
            branch_0=max_pool(var_input,ksize=[1,w,h,1],strides=[1,w,h,1],name='MaxPool_4x4_out')#[-1,4,4,out]
            branch_0=tf.reshape(branch_0,[-1,4*4*int(var_shape[3])],name='4x4_reshape')
            
        with tf.variable_scope('SPP2x2'):
            w=(var_shape[1]+2-1)//2  #2*2 maxpool with round up
            h=(var_shape[2]+2-1)//2
            branch_1=max_pool(var_input,ksize=[1,w,h,1],strides=[1,w,h,1],name='MaxPool_2x2_out')#[-1,2,2,out]
            branch_1=tf.reshape(branch_1,[-1,2*2*int(var_shape[3])],name='2x2_reshape')
        with tf.variable_scope('SPP1x1'):
            w=var_shape[1]  #1*1 maxpool with round up
            h=var_shape[2]
            branch_2=max_pool(var_input,ksize=[1,w,h,1],strides=[1,w,h,1],name='MaxPool_1x1_out')#[-1,1,1,out]
            branch_2=tf.reshape(branch_2,[-1,1*1*int(var_shape[3])],name='1x1_reshape')
        
        with tf.variable_scope('SPP6x8'):
            w=(var_shape[1]+6-1)//6  #4*4 maxpool with round up
            h=(var_shape[2]+8-1)//8
            branch_3=max_pool(var_input,ksize=[1,w,h,1],strides=[1,w,h,1],name='MaxPool_6x8_out')#[-1,6,8,out]
            branch_3=tf.reshape(branch_3,[-1,6*8*int(var_shape[3])],name='6x8_reshape')
        
        
        Concat_Temp=tf.concat([branch_0,branch_1,branch_2,branch_3],1)
        #bn=ops.batch_norm_layer(Concat_Temp,is_train=is_train,name='batch_norm_bn')
    return lrelu(Concat_Temp)#[-1,21x256] 5376

def Linear_Layer(var_input,name='Linear',is_train=True,keep_prob=1):
    with tf.variable_scope(name):
        linear1=linear_layer(var_input,1000,name='d_linear_layer1')#[-1,1000]
        linear1_prob = tf.nn.dropout(linear1, keep_prob)  
        
        #bn=ops.batch_norm_layer(linear1,is_train=is_train,name='batch_norm_bn1')
        #bn=group_norm_layer(linear1,50,1,0,name='d_bn_linear1')
        lr=lrelu(linear1_prob,name='d_lrelu_linear1')#[-1,1000]
        linear2=linear_layer(lr,500,name='d_linear_layer2')#[-1,500]
        #bn=bn=ops.batch_norm_layer(linear2,is_train=is_train,name='batch_norm_bn2')
        #bn=group_norm_layer(linear2,50,1,0,name='d_bn_linear2')
        lr=lrelu(linear2,name='d_lrelu_linear2')#[-1,500]    
        
        linear3=linear_layer(lr,100,name='d_linear_layer3')#[-1,3]   
        #bn=bn=ops.batch_norm_layer(linear3,is_train=is_train,name='batch_norm_bn3')
        lr=lrelu(linear3,name='d_lrelu_linear3')#[-1,2]
        
        linear4=linear_layer(lr,3,name='d_linear_layer4')#[-1,3]   
        #bn=bn=ops.batch_norm_layer(linear4,is_train=is_train,name='batch_norm_bn4')
        lr=lrelu(linear4,name='d_lrelu_linear4')#[-1,2]
        SoftMax=5*tf.nn.softmax(lr)#0~4, sigmoid 0.5~0.9933 1/(1+exp(-5))
    return SoftMax


def group_norm_layer(value,G,gamma,beta,name='group_norm',eps=1e-5): #value [N,h,w,C]
    value_size=np.size(value.get_shape())
    if (value_size==4):
        N=value.shape[0].value
        H=value.shape[1].value
        W=value.shape[2].value
        C=value.shape[3].value
    elif(value_size==2):
        N=value.shape[0].value
        C=value.shape[1].value
    else:
        print('input size neither 4 nor 2: %d'%(value_size))                       
    C_R=C%G
    with tf.variable_scope(name) as scope:
        if(C//G==0):
            print('Group too large:G: %d, C:%d'%(G,C))
        if(C_R):
            C_INT=C-C_R
              
            if(value_size==2):
                value_INT=value[:,0:C_INT]
                value_R=value[:,C_INT:C] 
                value_INT=tf.reshape(value_INT,[-1,C_INT//G,G])
                mean,var=tf.nn.moments(value_INT,[1],keep_dims=True)
                value_INT=(value_INT-mean)/tf.sqrt(var+eps)
                value_INT=tf.reshape(value_INT,[-1,C_INT])
                
                mean,var=tf.nn.moments(value_R,[1],keep_dims=True)
                value_R=(value_R-mean)/tf.sqrt(var+eps)
                value_R=tf.reshape(value_R,[-1,C_R])
                value=tf.concat([value_INT,value_R],1)
            elif(value_size==4):
                value_INT=value[:,:,:,0:C_INT]
                value_R=value[:,:,:,C_INT:C]
                print('C_INT:%d'%C_INT)
                print('value_int %d'%value_INT.shape[3])
                value_INT=tf.reshape(value_INT,[-1,H,W,C_INT//G,G])#!N
                mean,var=tf.nn.moments(value_INT,[1,2,3],keep_dims=True)             
                value_INT=(value_INT-mean)/tf.sqrt(var+eps)
                value_INT=tf.reshape(value_INT,[-1,H,W,C_INT])#!#
                mean,var=tf.nn.moments(value_R,[1,2,3],keep_dims=True)
                value_R=(value_R-mean)/tf.sqrt(var+eps)
                value_R=tf.reshape(value_R,[-1,H,W,C_R])##!N
                value=tf.concat([value_INT,value_R],3)          

        else:
            if(value_size==4):
                value=tf.reshape(value,[-1,H,W,C//G,G])#!N
                mean,var=tf.nn.moments(value,[1,2,3],keep_dims=True)
                value=(value-mean)/tf.sqrt(var+eps)
                value=tf.reshape(value,[-1,H,W,C])#!N
            elif(value_size==2):
                value=tf.reshape(value,[-1,C//G,G])#!N
                mean,var=tf.nn.moments(value,[1],keep_dims=True)
                value=(value-mean)/tf.sqrt(var+eps)
                value=tf.reshape(value,[-1,C])#!N
            
        return value*gamma+beta

 
def lrelu(x,leak=0,name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x,x*leak,name=name)
def one_hot(x,n):
    #x=np.array(x)
    assert x.ndim==1
    return np.eye(n)[x]

def SoftMax(var_input):
    length=len(var_input)
    var_out=np.array([[]])
    for i in range(length):
        Sum_Soft=np.sum(np.exp(var_input[i]))
        temp_out=np.exp(var_input[i])/Sum_Soft
        if math.isnan(temp_out[0]) or math.isnan(temp_out[1]):
           if var_input[i][0] or var_input[i][1]>10000000:
               var_input[i]=var_input[i]/10000
           c=np.exp(var_input[i]/np.log2(np.abs(var_input[i]))**2)
           Sum_Soft=np.sum(c)
           temp_out=c/Sum_Soft 
        if(var_out.shape[1]==0):#expeand dimension  
            var_out=np.append(var_out,[temp_out],axis=1)
        else:         
            var_out=np.append(var_out,[temp_out],axis=0)
    return var_out

def select97(SoftMax):
    length=len(SoftMax)
    var_out=np.array([])
    for i in range(length):
        out=SoftMax[i][1]
        if(out<0.97 and out>=0.5):
            out=0
        else:
            pass
        var_out=np.append(var_out,out)
    return var_out

def Accu_MovingAvergae(Accu,Shadow_Accu,decay=0.9):
    shadow_Accu=decay*Shadow_Accu+(1-decay)*Accu
    return shadow_Accu
        
    
    

