import tensorflow as tf
import numpy as np
import ops



class Network:
    def __init__(self):
        pass
    
    def NetGraph1(self,image,is_train=True,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        conv1=ops.Conv_Layer0(image,var_output=32,name='d_Conv_Layer1',is_train=is_train)#[48,64,32]
        gn=ops.group_norm_layer(conv1,5,1,0,name='d_gn_1')
        
        conv1_incep=ops.Cell_Incep1(gn,name='d_Cell_Inception1',is_train=is_train)#[-1,32,32,16*6=96]
        conv1_out=ops.Conv_Layer0(conv1_incep,var_output=32,name='d_Conv_Layer1_out',is_train=is_train)#[48,64,32]
        Add1=tf.add_n([gn,conv1_out])
        tf.add_to_collection('Add1',Add1)
        
        conv2=ops.Conv_Layer0(Add1,var_output=48,name='d_Conv_Layer2',is_train=is_train)#[48,64,48]
        conv2_incep=ops.Cell_Incep2(conv2,name='d_Cell_Inception2',is_train=is_train)#[-1,32,32,16*6=96]
        conv2_out=ops.Conv_Layer0(conv2_incep,var_output=48,name='d_Conv_Layer2_out',is_train=is_train)#[48,64,48]
        Add2=tf.add_n([conv2,conv2_out])
        
        
        conv2_1=ops.Conv_Layer0(Add2,var_output=32,name='d_Conv_Layer2_1',is_train=is_train)#[48,64,32]
        conv2_1_incep=ops.Cell_Incep3(conv2_1,name='d_Cell_Inception2_1',is_train=is_train)#[-1,32,32,16*6=48]
        conv2_1_out=ops.Conv_Layer0(conv2_1_incep,var_output=32,name='d_Conv_Layer2_1_out',is_train=is_train)#[48,64,32]
        Add2_1=tf.add_n([conv2_1,conv2_1_out])
        
        
        conv3=ops.Conv_Layer1(Add2_1,var_output=32,name='d_Conv_Layer3',is_train=is_train)#[24,32,32]
        conv3_incep=ops.Cell_Incep1(conv3,name='d_Cell_Inception3',is_train=is_train)#[-1,-1,-116*6=96]
        conv3_out=ops.Conv_Layer0(conv3_incep,var_output=32,name='d_Conv_Layer3_out',is_train=is_train)#[24,32,32]
        Add3=tf.add_n([conv3,conv3_out])
        
        conv4=ops.Conv_Layer0_1(Add3,var_output=32,name='d_Conv_Layer4',is_train=is_train)#[24,32,32]
        conv4_incep=ops.Cell_Incep2(conv4,name='d_Cell_Inception4',is_train=is_train)#[-1,-1,-1,16*6=96]
        conv4_out=ops.Conv_Layer0_1(conv4_incep,var_output=32,name='d_Conv_Layer4_out',is_train=is_train)#[24,32,32]
        Add4=tf.add_n([conv4,conv4_out])#[24,32,32]
        
        conv5=ops.Conv_Layer0_1(Add4,var_output=32,name='d_Conv_Layer5',is_train=is_train)#[24,32,16]
        conv5_incep=ops.Cell_Incep3(conv5,name='d_Cell_Inception5',is_train=is_train)#[-1,-1,-1,8*6=48]
        conv5_out=ops.Conv_Layer0_1(conv5_incep,var_output=32,name='d_Conv_Layer5_out',is_train=is_train)#[24,32,16]
        Add5=tf.add_n([conv5,conv5_out])#[24,32,16]
        
        
        conv6=ops.Conv_Layer2(Add5,var_output=32,name='d_Conv_Layer6',is_train=is_train)#[12,18,24]
        conv6_incep=ops.Cell_Incep3(conv6,name='d_Cell_Inception6',is_train=is_train)#[-1,-1,-1,8*6=48]
        conv6_out=ops.Conv_Layer0_1(conv6_incep,var_output=32,name='d_Conv_Layer6_out',is_train=is_train)#[12,18,24]
        Add6=tf.add_n([conv6,conv6_out])#[12,18,32]
        
        #SPP_out=ops.SPP(Add6,name='d_SPP',is_train=True)
        var_shapes=np.shape(Add6)#if add SPP, delete these two lines
        print(var_shapes)
        tf.add_to_collection('conv_output',conv6_out)
        conv6_re=tf.reshape(Add6,[-1,var_shapes[1]*var_shapes[2]*int(var_shapes[3])],name='var_reshape')
        #conv4_re=tf.reshape(conv4_incep,[-1,var_shapes[1]*var_shapes[2]*int(var_shapes[3])],name='var_reshape')
        #spp_layer=ops.SPP(conv3_incep,'d_SPP',is_train=is_train)
           
        linear_layer=ops.Linear_Layer(conv6_re,'d_linear_layer',is_train=is_train)
        return linear_layer
        
    def NetGraph(self,image,is_train=True,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        Incep1=ops.Cell_Incep1(image,name='Cell_Inception1',is_train=is_train)    
        conv1=ops.Conv_Layer1(Incep1,name='d_Conv_Layer1',var_output=32,is_train=is_train)
        conv1_1=ops.Conv_Layer1(Incep1,name='d_Conv_Layer1_1',var_output=32,is_train=is_train)
        conv1_out=tf.concat([conv1,conv1_1],3)
        
        block1=ops.Block(conv1_out,name='d_block1',inout_node=16,is_train=is_train)
        Trans1=ops.Transition(block1,w=5,h=5,var_output=16,name='Trans1',is_train=is_train)
        block2=ops.Block(Trans1,name='d_block2',inout_node=32,is_train=is_train)
        Trans2=ops.Transition(block2,w=3,h=3,name='Trans2',var_output=16,is_train=is_train)
        block3=ops.Block(Trans2,name='d_block3',inout_node=16,is_train=is_train)
        Trans3=ops.Transition(block3,w=3,h=3,name='Trans3',var_output=16,is_train=is_train)
        #block1=ops.Block(block2,name='d_block1',inout_node=inout_node,is_train=True)
        Trans_Con=tf.concat([Trans2,Trans1,Trans3],3) #[-1,24,32,48]
        
        Incep2=ops.Cell_Incep1(Trans_Con,name='Cell_Inception2',is_train=is_train)       
        conv2 =ops.Conv_Layer2(Incep2,name='d_Conv_Layer2',var_output=16,is_train=is_train)#[-1,12,16,16]
        conv2_1 =ops.Conv_Layer1(Incep2,name='d_Conv_Layer2_1',var_output=16,is_train=is_train)
        conv2_out=tf.add_n([conv2,conv2_1])
        block4=ops.Block(conv2_out,name='d_block4',inout_node=16,is_train=is_train)
        Trans4=ops.Transition(block4,w=5,h=5,var_output=16,name='Trans4',is_train=is_train)
        block5=ops.Block(Trans4,name='d_block5',inout_node=32,is_train=is_train)
        Trans5=ops.Transition(block5,w=3,h=3,var_output=16,name='Trans5',is_train=is_train)
        block6=ops.Block(Trans5,name='d_block6',inout_node=16,is_train=is_train)
        Trans6=ops.Transition(block6,w=3,h=3,var_output=16,name='Trans6',is_train=is_train)
        #block1=ops.Block(block2,name='d_block1',inout_node=inout_node,is_train=True)
        Trans_Con=tf.concat([Trans4,Trans5,Trans6],3) #[-1,12,16,48]
        conv_output=Trans_Con
        
        
        tf.add_to_collection('conv_output',conv_output)
        var_shapes=np.shape(conv_output)
        print(var_shapes)
        conv_re=tf.reshape(conv_output,[-1,var_shapes[1]*var_shapes[2]*var_shapes[3]],name='var_reshape')
        linear_layer=ops.Linear_Layer(conv_re,'d_linear_layer',is_train=is_train)
        return linear_layer
        
    #input [-1,240,320,3]        
    def NetGraph2(self,image,is_train=True,reuse=False,keep_prob=1,Res_Rate=0.5):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            gn=ops.group_norm_layer(image,3,1,0,name='d_gn_1')
            tf.add_to_collection('gn',gn)
            #STEM
            Filter_Concat=ops.input_Conv_Layer(gn,var_in=32,name='d_input_Conv_Layer1',is_train=is_train)#[-1,30,40,384]
            Net_Out=ops.lrelu(Filter_Concat)#[-1,30,40,384]
            
            Incep_Res_A=ops.Inception_ResNet_A(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_A1',is_train=is_train)#[-1,30,40,384]
            Net_Out=ops.lrelu(Incep_Res_A)#[-1,30,40,384]
            
            Incep_Res_A=ops.Inception_ResNet_A(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_A2',is_train=is_train)#[-1,30,40,384]
            Net_Out=ops.lrelu(Incep_Res_A)#[-1,30,40,384]
            
            Incep_Res_A=ops.Inception_ResNet_A(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_A3',is_train=is_train)#[-1,30,40,384]
            Net_Out=ops.lrelu(Incep_Res_A)#[-1,30,40,384]
            
            Incep_Res_A=ops.Inception_ResNet_A(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_A4',is_train=is_train)#[-1,30,40,384]
            Net_Out=ops.lrelu(Incep_Res_A)#[-1,30,40,384]
            
            Incep_Res_A=ops.Inception_ResNet_A(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_A5',is_train=is_train)#[-1,30,40,384]
            Net_Out=ops.lrelu(Incep_Res_A)#[-1,30,40,384]
            #Reduction A 
            Reduction_A1=ops.Reduction_A(Net_Out,var_in=32,name='Reduction_A1',is_train=True)
            Net_Out=ops.lrelu(Reduction_A1)#[-1,30,40,384]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B1',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B2',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B3',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B4',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B5',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B6',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B7',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B8',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B9',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            Incep_Res_B=ops.Inception_ResNet_B(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_B10',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_B)#[-1,30,40,1152]
            
            tf.add_to_collection('Inception_B',Net_Out)
            
            Reduction_B1=ops.Reduction_B(Net_Out,var_in=32,name='Reduction_B1',is_train=is_train)
            Net_Out=ops.lrelu(Reduction_B1)
            
            Incep_Res_C=ops.Inception_ResNet_C(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_C1',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_C)#[-1,8,10,2144]
            
            Incep_Res_C=ops.Inception_ResNet_C(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_C2',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_C)
            
            Incep_Res_C=ops.Inception_ResNet_C(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_C3',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_C)
            
            Incep_Res_C=ops.Inception_ResNet_C(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_C4',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_C)
            
            Incep_Res_C=ops.Inception_ResNet_C(Net_Out,var_in=32,Res_Rate=Res_Rate,name='Inception_ResNet_C5',is_train=is_train)
            Net_Out=ops.lrelu(Incep_Res_C)#[-1,8,10,2144]
            
            Sel=ops.Selection(Net_Out,var_out=500,name='Selection_Layer',is_train=True)
            Net_Out=ops.lrelu(Sel)
            
            Average_Layer=ops.average_pool(Net_Out,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='Avg_Pool')#[-1,8,10,2144]
            
            Drop_Layer = tf.nn.dropout(Average_Layer, keep_prob)
            
            
            #conv4_re=tf.reshape(conv4_incep,[-1,var_shapes[1]*var_shapes[2]*int(var_shapes[3])],name='var_reshape')
            #spp_layer=ops.SPP(conv3_incep,'d_SPP',is_train=is_train)
            var_shapes=np.shape(Drop_Layer)
            conv_re=tf.reshape(Drop_Layer,[-1,var_shapes[1]*var_shapes[2]*var_shapes[3]],name='var_reshape') 
            linear_layer=ops.Linear_Layer_For_Incep_Res(conv_re,'d_linear_layer',is_train=is_train,keep_prob=keep_prob)
            #then softmax
            
            return linear_layer           
        
        

