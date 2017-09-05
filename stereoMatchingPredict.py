

import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import scipy.misc 
import scipy.io

ID = 2500
M = 3

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
Dir = os.getcwd() # change path 
MODEL_FILENAME = Dir+"/tensorflow_workstation/anita/7_layer_tv/kitti_7layer_2_300000"
#MODEL_FILENAME = Dir+"/tensorflow_workstation/anita/9_layer_3pool/kitti_9layer_3pool_1200000"
#MODEL_FILENAME = Dir+"/tensorflow_workstation/anita/7_layer_nopooling/7layer_nopooling_4"


""" ---------- Helper functions ------------- """

def disp_image_to_label(disp_image,nclasses):
    disp_image[disp_image>nclasses-1]=0
    return disp_image



def load_kitti():
    print('load_images:')

    left_images_validation = np.load('tensorflow_workstation/anita/colon/left_image_f'+str(ID)+'.npy')
    right_images_validation = np.load('tensorflow_workstation/anita/colon/right_image_f'+str(ID)+'.npy')
    return left_images_validation, right_images_validation

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

def network_7l_2p(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu_pool(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu(h2, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv3')
    h4 = conv_relu_pool(h3, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    h5 = conv_relu(h4, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv5')
    h6 = conv_relu(h5, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7 = conv(h6, [3, 3, n_units, n_units], [n_units],reuse,'conv7')
    h8 = deconv(h7, [3, 3, n_units, n_units], [n_units], [batch_size,tf.cast(imSize_h/2,tf.int32),tf.cast(imSize_w/2,tf.int32),n_units],reuse,'conv8')
    h9 = deconv(h8, [3, 3, n_units, n_units], [n_units], [batch_size,imSize_h,imSize_w + disp,n_units],reuse,'conv9')   
    return h9

def network_9l_2p(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu_pool(h2, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv3')
    h4 = conv_relu(h3, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    h5 = conv_relu(h4, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv5')
    h6 = conv_relu_pool(h5, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7 = conv_relu(h6, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv7')
    h8 = conv_relu(h7, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv8')
    h9 = conv(h8, [3, 3, n_units, n_units], [n_units],reuse,'conv9')
    h10 = deconv(h9, [3, 3, n_units, n_units], [n_units], [batch_size,tf.cast(imSize_h/2,tf.int32),tf.cast(imSize_w/2,tf.int32),n_units],reuse,'conv10')
    h11 = deconv(h10, [3, 3, n_units, n_units], [n_units], [batch_size,imSize_h,imSize_w + disp,n_units],reuse,'conv11')   
    return h11

def network_7l_0p(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu(h2, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv3')
    h4 = conv_relu(h3, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    h5 = conv_relu(h4, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv5')
    h6 = conv_relu(h5, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7 = conv(h6, [3, 3, n_units, n_units], [n_units],reuse,'conv7')
    return h7

def network_9l_0p(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu(h2, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv3')
    h4 = conv_relu(h3, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    h5 = conv_relu(h4, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv5')
    h6 = conv_relu(h5, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7 = conv_relu(h6, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv7')
    h8 = conv_relu(h7, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv8')
    h9 = conv(h8, [3, 3, n_units, n_units], [n_units],reuse,'conv9')
    return h9

def network_9l_3p(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu_pool(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu(h2, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv3')
    h4 = conv_relu_pool(h3, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv4')
    h5 = conv_relu(h4, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv5')
    h6 = conv_relu_pool(h5, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv6')
    h7 = conv_relu(h6, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv7')
    h8 = conv_relu(h7, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv8')
    h9 = conv(h8, [3, 3, n_units, n_units], [n_units],reuse,'conv9')
    h10 = deconv(h9, [3, 3, n_units, n_units], [n_units], [batch_size,tf.cast(imSize_h/4,tf.int32),tf.cast(imSize_w/4,tf.int32),n_units],reuse,'conv10')
    h11 = deconv(h10, [3, 3, n_units, n_units], [n_units], [batch_size,tf.cast(imSize_h/2,tf.int32),tf.cast(imSize_w/2,tf.int32),n_units],reuse,'conv11')   
    h12 = deconv(h11, [3, 3, n_units, n_units], [n_units], [batch_size,imSize_h,imSize_w + disp,n_units],reuse,'conv12')
    return h12

def network_4l_1p(input,reuse,disp,phase):
    h1 = conv_relu(input, [3, 3, n_channels, n_units], [n_units],phase,reuse, 'conv1')
    h2 = conv_relu_pool(h1, [3, 3, n_units, n_units], [n_units],phase,reuse, 'conv2')
    h3 = conv_relu(h2, [3, 3, n_units, n_units], [n_units],phase,reuse,'conv3')
    h4 = conv(h3, [3, 3, n_units, n_units], [n_units],reuse,'conv4')
    h5 = deconv(h4, [3, 3, n_units, n_units], [n_units], [batch_size,imSize_h,imSize_w + disp,n_units],reuse,'conv5')   
    return h5

    
def make_graph(x_left,x_right,y_,maxDisp,learning_rate,global_step,phase):
    with tf.variable_scope("siamese_network") as scope:
        h9_left = network_7l_2p(input = x_left,reuse= False,disp = 0,phase=phase)
        h9_right = network_7l_2p(input = x_right,reuse = True,disp = 0,phase=phase)

    return h9_left,h9_right

def init_placeholders(imSize_h,imSize_w, n_channels,maxDisp):
    phase = tf.placeholder(tf.bool)
    
    with tf.name_scope('ImLeft'):
        x_left = tf.placeholder(tf.float32, shape=[None, imSize_h, imSize_w, n_channels])
    
    with tf.name_scope('ImRight'):    
        x_right = tf.placeholder(tf.float32, shape=[None, imSize_h, imSize_w, n_channels])
    
    with tf.name_scope('Labels'):
        y_ = tf.placeholder(tf.int32, shape=[None,imSize_h,imSize_w])
    return phase,x_left,x_right,y_
    
def error(output,y_,threshold=3):
    errors = np.abs(output-y_);
    valid_pixels = errors[y_!=0]
    n_err   = np.sum(valid_pixels > threshold);
    n_total = len(valid_pixels);
    return float(n_err)/float(n_total)

    

def inner_product_test(h9_left, h9_right,batch_size,rows,cols, n_classes):
    prod=np.ones((batch_size,rows,cols,n_classes))*(-1e9)
    start=0
    end = cols

    while start<cols-1:
        for disp in range(n_classes):
            if (end-disp  > 0):
                if (cols > start-disp ):

                    left_features = h9_left[:,:,max(start,disp):min(end,cols),:]
                    right_features = h9_right[:,:,max(0,start-disp ):min(end-disp ,cols-disp),:]
                    
                    multiplication = np.multiply(left_features,right_features)
                    inner_product = np.sum(multiplication,axis=3)
                    prod[:,:,max(start,disp):min(end,cols),disp]=inner_product
        
        start = end
        end += cols

    return prod



def get_loss(y_,output_layer):
    valid_labels=tf.not_equal(y_, 0)            
    y_bl=tf.boolean_mask(y_, valid_labels)            
    logits = tf.boolean_mask(output_layer, valid_labels)
    logits = tf.reshape(logits, [-1,n_classes])
    y_bl = tf.reshape(y_bl, [-1])
    return logits, y_bl
""" -------------  main ------------------- """
def test_model():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,400, 0.994, staircase=True)
      
    left_images, right_images = load_kitti()
    
    phase,x_left,x_right,y_ = init_placeholders(imSize_h,imSize_w, n_channels,maxDisp)

    h9_left,h9_right = make_graph(x_left,x_right,y_,maxDisp,learning_rate,global_step,phase)   

     
    config = tf.ConfigProto()
    
    #run session
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    #summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    saver.restore(sess = sess, save_path= MODEL_FILENAME)
    print('Model loaded.')
    im_gt = []
    image_left = []
    image_right = []
    total_error = 0
    for j in range(1):
        print(j)

        left = left_images[np.newaxis,...]
        right = right_images[np.newaxis,...]
        #print(left.shape)
        left_resized = tf.image.resize_images(left,resize_shape)
        right_resized = tf.image.resize_images(right, resize_shape)
        
        #print(left_resized)
        left_,right_ = sess.run([left_resized,right_resized])
        #print('test')
        #im_gt = np.transpose(gt_,[2,0,1])
 

      
        h9_left_,h9_right_ = sess.run([h9_left,h9_right], feed_dict={x_left:left_,
                                          x_right:right_,
                                          phase:False})
    #print(h9_left_)

        test = inner_product_test(h9_left_,h9_right_,batch_size,500,640,129)
    #print(test)
        print(np.shape(test))
    #out0 = np.transpose(test,[1,2,3,0]) 
        softmax = tf.nn.softmax(test)
        softmax_ = sess.run([softmax])
    #logits_, y_bl_ = get_loss(y_,output_layer_t) --> must go into placeholder as well 
    #print(output_layer_t_.shape())

        tstim = np.argmax(test,3)[0,:,:]
        #tstidx = im_gt[0,:,:]==0
        #tstim[tstidx] = 0
        #print(np.shape(tstim))
        #error_val_1=error(np.argmax(test,3),im_gt)
        plt.imshow(tstim)
        #plt.savefig("tensorflow_workstation/anita/disparity_colon_test1.png")
        #plt.show()
        img = scipy.misc.toimage(tstim, high=np.max(tstim), low=np.min(tstim), mode='I')
        img.save('tensorflow_workstation/anita/porcine/allf/p_disp_f'+str(ID)+'_M'+str(M)+'test.png')
        print(np.shape(softmax_))
        tst_softmax = np.squeeze(np.max(softmax_,-1))
        volume = np.squeeze(softmax_)
        print(np.shape(tst_softmax))
        print(tst_softmax)
        #plt.imshow(tst_softmax)
        #plt.show()
        #scipy.io.savemat('tensorflow_workstation/anita/porcine/allf/p_probs_f'+str(ID)+'_M'+str(M)+'.mat',{"certainties":tst_softmax})
        #scipy.io.savemat('tensorflow_workstation/anita/porcine/allf/p_vol_f'+str(ID)+'_M'+str(M)+'.mat',{"certain_volume":volume})


if __name__ == '__main__':
    np.random.seed(seed=123) 
    n_units = 64
    maxDisp = 128
    batch_size = 1
    n_classes = maxDisp + 1
    n_channels = 3
    receptive_field = 28
    right_left = 28
    right_size = right_left+maxDisp
    resize_shape=[500,640]
    imSize_h = resize_shape[0]
    imSize_w = resize_shape[1]   
    test_model()


