
# coding: utf-8

# In[8]:

import numpy as np
import tensorflow as tf
import os
from scipy import ndimage
from matplotlib import pyplot as plt


# In[19]:

# specify directory
Dir = "C:/Users/Anita/Documents/UCL/Dissertation/code/" # CHANGE THIS, please provide full path
# test if directory works
if not os.path.exists(Dir):
    print("Please change directory")
logs_path = 'trained_models_3'
MODEL_FILENAME = Dir+"blaze/patches_saved_10.ckpt"
MODEL_FILENAME_2 = Dir+"saved_models_5/patches_saved.ckpt"


# In[21]:

# helper functions
def disp_image_to_label(disp_image,nclasses):
    disp_image[disp_image>nclasses-1]=0
    return disp_image

def get_valid_pixels(training_list_noc_label,total_patch_size,maxDisp):
    half_path_size=int(total_patch_size/2)
    half_max_disp=int(maxDisp/2)
    list_of_valid_pixels=[]
    for i in range(len(training_list_noc_label)):
        valid_choices=np.where(training_list_noc_label[i]!=0)

        #put this in a funciton or something
        #transpose
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
    #half_receptive=int(patch_size/2)
    half_max_disp=int(maxDisp/2)


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


# In[24]:

#batch_left


# In[4]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def conv_relu(input, kernel_shape, bias_shape, phase,reuse,scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights = tf.get_variable("weights", kernel_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d()) #xavier
        biases = tf.get_variable("biases", bias_shape, initializer=tf.contrib.layers.xavier_initializer())
        conv = conv2d(input, weights)
        #normal = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv,biases), 
                                          #center=True, scale=True, 
                                          #is_training=phase, decay=0.9,
                                          #scope='bn')
        normal = tf.nn.bias_add(conv,biases)
        return tf.nn.relu(conv),tf.nn.relu(normal)

def conv_relu_pool(input, kernel_shape, bias_shape,phase,reuse,scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights = tf.get_variable("weights", kernel_shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable("biases", bias_shape, initializer=tf.contrib.layers.xavier_initializer())
        conv = conv2d(input, weights)
        #normal = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv,biases), 
        #                                  center=True, scale=True, 
        #                                  is_training=phase, decay=0.9,
        #                                  scope='bn')
        normal = tf.nn.bias_add(conv,biases)
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

def network(input,reuse,disp):
    h1_bn, h1_ = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2_ = conv_relu_pool(h1_, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    _,h3_ = conv_relu(h2_, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv3')
    h4_ = conv_relu_pool(h3_, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    _,h5_ = conv_relu(h4_, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv5')
    _,h6_ = conv_relu(h5_, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7_ = conv(h6_, [3, 3, n_units, n_units], [n_units],reuse,'conv7')
    h8_ = deconv(h7_, [3, 3, n_units, n_units], [n_units], [batch_size,HimSize,tf.cast(imSize/2+(disp/2),tf.int32),n_units],reuse,'conv8')
    h9 = deconv(h8_, [3, 3, n_units, n_units], [n_units], [batch_size,imSize,imSize + disp,n_units],reuse,'conv9')
    return h1_bn, h1_, h9


# In[12]:

# define parameters 
learn_rate = 0.0001
n_units = 64

maxDisp = 128
batch_size = 2
n_classes = maxDisp + 1
n_channels = 3
receptive_field = 28#10#13#45#97
right_left = 28
right_size = right_left+maxDisp
imSize = receptive_field
HimSize = tf.cast(imSize/2,tf.int32)

print('load_images:')
left_images = np.load('left_images.npy')
right_images = np.load('right_images.npy')
disp_images = np.load('disp_images.npy')
left_images_validation = np.load('validation_list_left.npy')
right_images_validation = np.load('validation_list_right.npy')
disp_images_validation = np.load('validation_list_noc_label.npy')

print('process images:')
valid_pixels_train = get_valid_pixels(disp_images,receptive_field,maxDisp)
valid_pixels_val = get_valid_pixels(disp_images_validation,receptive_field,maxDisp)
np.save('valid_pixels_train',valid_pixels_train)
np.save('valid_pixels_val',valid_pixels_val)
#valid_pixels_train  = np.load('valid_pixels_train .npy')
#valid_pixels_val = np.load('valid_pixels_val.npy')
print('Valid pixels extracted!')


# In[13]:

## Build graph
with tf.name_scope('ImLeft'):
    x_left = tf.placeholder(tf.float32, shape=[None, imSize, imSize, n_channels])
with tf.name_scope('ImRight'):    
    x_right = tf.placeholder(tf.float32, shape=[None, imSize, imSize +maxDisp, n_channels])
with tf.name_scope('Labels'):
    y_ = tf.placeholder(tf.float32, shape=[None,imSize,imSize])
    
y_onehot = tf.one_hot(tf.cast(y_,tf.int32),n_classes,axis=3)    
phase = tf.placeholder(tf.bool)

with tf.variable_scope("siamese_network") as scope:
    h1_bn_l,h1_l,h9_left = network(input = x_left,reuse= False,disp = 0)
    _,_,h9_right = network(input = x_right,reuse = True,disp = maxDisp)


# In[14]:

#np.save('valid_pixels_train',valid_pixels_train)
#np.save('valid_pixels_val',valid_pixels_val)


# In[ ]:




# In[16]:

# multiplication layer
valid_n = h9_right.get_shape().as_list()[2]-h9_left.get_shape().as_list()[2]+1
multi = []
for i in range(valid_n):
    test = tf.reduce_sum(np.multiply(h9_left,h9_right[:,:,i:i+imSize,:]),3)
    multi.append(test)
with tf.name_scope('Model'):
    output_layer = tf.stack(multi)  


# In[17]:

# set up error and optimizer
output_layer_t = tf.transpose(output_layer,[1,2,3,0])
output = tf.reshape(output_layer_t,[batch_size*imSize*imSize,n_classes])
max_ = tf.argmax(output_layer_t[0,:,:,:],0)
pred = tf.nn.softmax(tf.reshape(output_layer_t[0,:,:,:],[imSize*imSize,n_classes]))
labs = tf.reshape(y_onehot[0,:,:,:],(1*imSize*imSize,n_classes))

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(y_onehot,(batch_size*imSize*imSize,n_classes)), logits = output))
    
    
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy) #Adam
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(tf.reshape(y_onehot,(batch_size*imSize*imSize,n_classes)),1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred2 = tf.argmax(output_layer,2)
    
# Create summary for loss and accuracy
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()



# In[22]:



#run session
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
#saver.restore(sess = sess, save_path= MODEL_FILENAME)
for i in range(200):
    print('begin training')
    #image_left,image_right,im_gt=load_random_patch(left_images,right_images,disp_images,receptive_field,right_left,maxDisp,batch_size,valid_pixels_train)
    #print(tf.get_collection(tf.GraphKeys.VARIABLES, scope="siamese_network"))
    training_list_left,training_list_right,training_list_noc_label,receptive_field = left_images,right_images,disp_images,receptive_field
    receptive_field_right,maxDisp,batch_size,valid_pixels_list = right_left,maxDisp,batch_size,valid_pixels_train
    total_patch_size=receptive_field
    half_path_size=int(total_patch_size/2)
    half_path_size_w=int(receptive_field_right/2)
    #half_receptive=int(patch_size/2)
    half_max_disp=int(maxDisp/2)


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

        image_left,image_right,im_gt = batch_left,batch_right,batch_disp
    print('step: ',i)    
    _,loss_,summary = sess.run([train_step,cross_entropy,merged_summary_op], feed_dict={x_left:image_left,
                                                                                        x_right:image_right,
                                                                                        y_:im_gt,
                                                                                        phase:True})
   
    summary_writer.add_summary(summary, i)
    if (i+1) % 2 == 0:
        print('------',i)
        
        max__,loss1,pred_,y_labs,h1_bn_l1,h1_l1 = sess.run([max_,cross_entropy,pred,labs,h1_bn_l,h1_l], feed_dict={x_left:image_left,
                                                                                        x_right:image_right,
                                                                                        y_:im_gt,
                                                                                        phase:True})
        max__,loss0,pred_,y_labs,h1_bn_l0,h1_l0 = sess.run([max_,cross_entropy,pred,labs,h1_bn_l,h1_l], feed_dict={x_left:image_left,
                                                                                        x_right:image_right,
                                                                                        y_:im_gt,
                                                                                        phase:False})
        print('training loss: ',loss1)
        print('testing loss: ',loss0)
        plt.imshow(np.argmax(pred_,1).reshape(28,28), interpolation='nearest')
        plt.show()
        plt.imshow(np.argmax(y_labs,1).reshape(28,28), interpolation='nearest')
        plt.show()
        print()
        print('training:')
        print('before norm: ', h1_bn_l1[1,8:20,8:20,1])
        print('after norm: ', h1_l1[1,8:20,8:20,1])
        print('testing:')
        print('before norm: ', h1_bn_l0[1,8:20,8:20,1])
        print('after norm: ', h1_l0[1,8:20,8:20,1])
saver.save(sess, MODEL_FILENAME_2)


# In[ ]:

# restore model and calculate testing accuracy and loss 
with tf.Session() as sess:
    idx = np.random.randint(160, size=batch_size)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess = sess, save_path= MODEL_FILENAME)
    print("Restored values:")
    #_,_,beta, gamma, mean, var,_,_,_,_,_,_,_,_ = [v.eval() for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="siamese_network/conv1/")]
        
    max__,acc_,pred_,y_labs,loss_,h1_bn_l0,h1_l0 = sess.run([max_,accuracy,pred,labs,cross_entropy,h1_bn_l,h1_l], feed_dict={x_left:trainImsLeft[idx,:,:].reshape(-1,imSize,imSize,1),
                                            x_right:trainImsRight[idx,:,:].reshape(-1,imSize,imSize+maxDisp,1),
                                            y_:trainLabs[idx,:,:].reshape(-1,imSize,imSize), phase:False})
    print(loss_)
    print(acc_)
    print('testing:')
    print('before norm: ', h1_bn_l0[:,1,1])
    print('after norm: ', h1_l0[:,1,1])
    plt.imshow(np.argmax(pred_,1).reshape(28,28), interpolation='nearest')
    plt.show()
    plt.imshow(np.argmax(y_labs,1).reshape(28,28), interpolation='nearest')
    plt.show()


# In[ ]:



