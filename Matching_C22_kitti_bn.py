# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:32:24 2017

@author: Anita
"""


import numpy as np
import tensorflow as tf
import os
#from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
Dir = os.getcwd() # change path 
if not os.path.exists(Dir):
    print("Please change directory")
logs_path = Dir + '/tensorflow_workstation/anita/trained_models_bn_22'
MODEL_FILENAME = Dir+"/tensorflow_workstation/anita/kitti_saved_bn_22.ckpt"


""" ---------- Patrick's helper functions ------------- """

def disp_image_to_label(disp_image,nclasses):
    disp_image[disp_image>nclasses-1]=0
    return disp_image

def get_valid_pixels(training_list_noc_label,total_patch_size,maxDisp):
    half_path_size=int(total_patch_size/2)
    half_max_disp=int(maxDisp/2)
    list_of_valid_pixels=[]
    for i in range(len(training_list_noc_label)):
        valid_choices=np.where(training_list_noc_label[i]!=0)
        valid_choices=list(map(list, zip(*valid_choices)))
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]>half_path_size]
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]<training_list_noc_label[i].shape[0]-half_path_size]
        #
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]>half_path_size+maxDisp+training_list_noc_label[i][x_y_pair[0]][x_y_pair[1]]]
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]<training_list_noc_label[i].shape[1]-half_max_disp-half_path_size]
        list_of_valid_pixels.append(valid_choices)
    return list_of_valid_pixels

def load_random_patch(training_list_left,training_list_right,training_list_noc_label,receptive_field,receptive_field_right,maxDisp,batch_size,
valid_pixels_list):
    total_patch_size=receptive_field
    half_path_size=int(total_patch_size/2)
    half_path_size_w=int(receptive_field_right/2)

    batch_left=np.zeros((0, total_patch_size,receptive_field_right,3), dtype=np.float32)
    batch_right=np.zeros((0, total_patch_size,maxDisp+receptive_field_right,3), dtype=np.float32)
    batch_disp=np.zeros((0, total_patch_size,receptive_field_right), dtype=np.float32)

    for batch in range(0,batch_size):
        random_image_index=np.random.randint(len(training_list_left))        
        random_choice=np.random.randint(len(valid_pixels_list[random_image_index]))
        x_rand,y_rand=valid_pixels_list[random_image_index][random_choice][0],valid_pixels_list[random_image_index][random_choice][1]
        #left=training_list_left[random_image_index][max(0,x_rand-half_path_size):x_rand+half_path_size+1,max(0,y_rand-half_path_size):y_rand+half_path_size+1,:]
        left=training_list_left[random_image_index][x_rand-half_path_size:x_rand+half_path_size,
        y_rand-half_path_size_w:y_rand+half_path_size_w,:]
        d=training_list_noc_label[random_image_index][x_rand-half_path_size:x_rand+half_path_size,y_rand-half_path_size:y_rand+half_path_size]
        right=training_list_right[random_image_index][x_rand-half_path_size:x_rand+half_path_size,y_rand-half_path_size-maxDisp:y_rand+half_path_size,:]
        # plt.figure(0)
        # plt.imshow(left)
        # plt.figure(1)
        # plt.imshow(right)
        # plt.figure(2)
        # plt.imshow(d,cmap="gray")
        # print(d)
        # plt.show()

        d=disp_image_to_label(d,maxDisp+1)

        left = left[np.newaxis,...]
        right = right[np.newaxis,...]
        d= d[np.newaxis,...]

        #print(d.shape,batch_disp.shape)
        batch_left=np.concatenate([batch_left, left],axis=0)
        batch_right=np.concatenate([batch_right, right],axis=0)
        batch_disp=np.concatenate([batch_disp, d],axis=0)

    return batch_left,batch_right,batch_disp

def load_kitti():
    print('load_images:')
    left_images = np.load('tensorflow_workstation/anita/left_images.npy')
    right_images = np.load('tensorflow_workstation/anita/right_images.npy')
    disp_images = np.load('tensorflow_workstation/anita/disp_images.npy')
    left_images_validation = np.load('tensorflow_workstation/anita/validation_list_left.npy')
    right_images_validation = np.load('tensorflow_workstation/anita/validation_list_right.npy')
    disp_images_validation = np.load('tensorflow_workstation/anita/validation_list_noc_label.npy')
    
    print('process images:')
    #valid_pixels_train = get_valid_pixels(disp_images,receptive_field,maxDisp)
    #valid_pixels_val = get_valid_pixels(disp_images_validation,receptive_field,maxDisp)
    #np.save('valid_pixels_train',valid_pixels_train)
    #np.save('valid_pixels_val',valid_pixels_val)
    valid_pixels_train  = np.load('tensorflow_workstation/anita/valid_pixels_train.npy')
    valid_pixels_val = np.load('tensorflow_workstation/anita/valid_pixels_val.npy')
    print('Valid pixels extracted!')
    return left_images, right_images, disp_images, left_images_validation, right_images_validation, disp_images_validation, valid_pixels_train, valid_pixels_val

""" -------------------------- functions to build graph ------------------------------------ """

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def conv_relu(input, kernel_shape, bias_shape, phase,reuse,scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights = tf.get_variable("weights", kernel_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d()) 
        biases = tf.get_variable("biases", bias_shape, initializer=tf.contrib.layers.xavier_initializer())
        conv = conv2d(input, weights)
        normal = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv,biases), 
                                          center=True, scale=True, 
                                          is_training=phase, decay=0.9,
                                          scope='bn')
        return tf.nn.relu(normal)

def conv_relu_pool(input, kernel_shape, bias_shape,phase,reuse,scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights = tf.get_variable("weights", kernel_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.contrib.layers.xavier_initializer())
        conv = conv2d(input, weights)
        
        normal = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv,biases), 
                                          center=True, scale=True, 
                                          is_training=phase, decay=0.9,
                                          scope='bn')
        return max_pool(tf.nn.relu(normal))

def conv(input, kernel_shape, bias_shape, reuse, scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights = tf.get_variable("weights", kernel_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.contrib.layers.xavier_initializer())
        conv = conv2d(input, weights)
        return tf.nn.bias_add(conv,biases)

def deconv(input, kernel_shape, bias_shape, output_shape, reuse, scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights = tf.get_variable("weights", kernel_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.contrib.layers.xavier_initializer())    
        conv = tf.nn.conv2d_transpose(input, weights,output_shape,[1,2,2,1])
        return tf.nn.bias_add(conv,biases)

def network(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu_pool(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu(h2, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv3')
    h4 = conv_relu_pool(h3, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    h5 = conv_relu(h4, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv5')
    h6 = conv_relu(h5, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7 = conv(h6, [3, 3, n_units, n_units], [n_units],reuse,'conv7')
    h8 = deconv(h7, [3, 3, n_units, n_units], [n_units], [batch_size,HimSize,tf.cast(imSize/2+(disp/2),tf.int32),n_units],reuse,'conv8')
    h9 = deconv(h8, [3, 3, n_units, n_units], [n_units], [batch_size,imSize,imSize + disp,n_units],reuse,'conv9')   
    return h9
    
def make_graph(x_left,x_right,y_,maxDisp,learning_rate,global_step,phase):
    with tf.variable_scope("siamese_network") as scope:
        h9_left = network(input = x_left,reuse= False,disp = 0,phase=phase)
        h9_right = network(input = x_right,reuse = True,disp = maxDisp,phase=phase)
    
    with tf.name_scope('Model'):
        output_layer_t = tf.transpose(inner_product(h9_left,h9_right),[1,2,3,0]) 
        
    logits, y_bl = get_loss(y_,output_layer_t)
    
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_bl, name=None)
    with tf.name_scope('Loss'):
        cross_entropy = tf.reduce_mean(loss)      
            
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step) #Adam
    return train_step, cross_entropy, output_layer_t

def init_placeholders(imSize, n_channels,maxDisp):
    phase = tf.placeholder(tf.bool)
    
    with tf.name_scope('ImLeft'):
        x_left = tf.placeholder(tf.float32, shape=[None, imSize, imSize, n_channels])
    
    with tf.name_scope('ImRight'):    
        x_right = tf.placeholder(tf.float32, shape=[None, imSize, imSize +maxDisp, n_channels])
    
    with tf.name_scope('Labels'):
        y_ = tf.placeholder(tf.int32, shape=[None,imSize,imSize])
    return phase,x_left,x_right,y_
    
def error(output,y_,threshold=3):
    errors = np.abs(output-y_);
    valid_pixels = errors[y_!=0]
    n_err   = np.sum(valid_pixels > threshold);
    n_total = len(valid_pixels);
    return float(n_err)/float(n_total)

def inner_product(h9_left,h9_right):
    valid_n = h9_right.get_shape().as_list()[2]-h9_left.get_shape().as_list()[2]+1
    multi = []
    for i in range(valid_n):
        prod = tf.reduce_sum(np.multiply(h9_left,h9_right[:,:,valid_n - i -1:valid_n - i - 1+imSize,:]),3)
        multi.append(prod)
    return tf.stack(multi)

def get_loss(y_,output_layer):
    valid_labels=tf.not_equal(y_, 0)            
    y_bl=tf.boolean_mask(y_, valid_labels)            
    logits = tf.boolean_mask(output_layer, valid_labels)
    logits = tf.reshape(logits, [-1,n_classes])
    y_bl = tf.reshape(y_bl, [-1])
    return logits, y_bl
""" ------------- should go into main ------------------- """
def train_model():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,200, 0.994, staircase=True)
      
    left_images, right_images, disp_images, left_images_validation, right_images_validation, disp_images_validation, valid_pixels_train, valid_pixels_val = load_kitti()
    
    phase,x_left,x_right,y_ = init_placeholders(imSize, n_channels,maxDisp)

    train_step, cross_entropy, output_layer_t = make_graph(x_left,x_right,y_,maxDisp,learning_rate,global_step,phase)   
    accuracy=0
        
    # Create summary for loss and accuracy
    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()        
    config = tf.ConfigProto()
    
    #run session
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    #saver.restore(sess = sess, save_path= MODEL_FILENAME_2)
    print('begin training')
    for i in range(150000):
        
        image_left,image_right,im_gt=load_random_patch(left_images,right_images,disp_images,receptive_field,right_left,maxDisp,batch_size,valid_pixels_train)
        #print(tf.get_collection(tf.GraphKeys.VARIABLES, scope="siamese_network"))
   
        _,summary = sess.run([train_step,merged_summary_op], feed_dict={x_left:image_left,
                             x_right:image_right,
                             y_:im_gt,
                             phase:True})
       
        summary_writer.add_summary(summary, i)
        if (i+1) % 100 == 0:
            print('------',i+1)
        
            loss1, out1 = sess.run([cross_entropy,output_layer_t], feed_dict={x_left:image_left,
                                          x_right:image_right,
                                          y_:im_gt,
                                          phase:True})
            loss0, out0 = sess.run([cross_entropy,output_layer_t], feed_dict={x_left:image_left,
                                          x_right:image_right,
                                          y_:im_gt,
                                          phase:False})
            error_val_1=error(np.argmax(out0,3),im_gt)
    
            print('training loss: ',loss1)
            print('testing loss: ',loss0, 'testing error: ', error_val_1)
    
            saver.save(sess, MODEL_FILENAME)

# some global variables
if __name__ == '__main__':
    np.random.seed(seed=123) 
    n_units = 64
    maxDisp = 128
    batch_size = 32
    n_classes = maxDisp + 1
    n_channels = 3
    receptive_field = 28
    right_left = 28
    right_size = right_left+maxDisp
    imSize = receptive_field
    HimSize = tf.cast(imSize/2,tf.int32)
    train_model()


