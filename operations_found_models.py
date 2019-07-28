
import tensorflow as tf
import tensorflow.contrib.slim as slim

NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?

# batch norm
def batch_norm(input_, name="bn_in", decay=0.1, scale=True, is_training=True, epsilon=1e-4):
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        moving_mean = tf.get_variable('moving_mean',shape[-1], initializer=tf.constant_initializer(0), trainable=False)
        moving_var = tf.get_variable('moving_var',shape[-1], initializer=tf.constant_initializer(1.), trainable=False)
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.))
        if scale:
            gamma = tf.get_variable('gamma', shape[-1], initializer=tf.constant_initializer(1.))
        else:
            gamma = tf.get_variable('gamma', shape[-1], initializer=tf.constant_initializer(1.), trainable=False)

        if is_training:
            out, cur_mean, cur_var = tf.nn.fused_batch_norm(input_, gamma, beta
                                                            , epsilon=epsilon, is_training=is_training)
            #todo check for nan?  tf.check_numerics
            update_mean_op = moving_mean.assign_sub(decay*(moving_mean-cur_mean))
            update_var_op = moving_var.assign_sub(decay*(moving_var-cur_var))
            with tf.control_dependencies([update_mean_op,update_var_op]):
                out = tf.identity(out)
        else:
            out, _, _ = tf.nn.fused_batch_norm(input_, gamma, beta,
                                               moving_mean, moving_var, epsilon=epsilon, is_training=is_training)
    return out

def cond_batchnorm(inputs,name,is_training, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    shape = inputs.get_shape().as_list()
    offset = tf.constant(0,tf.float32,shape=[shape[3]])
    scale = tf.constant(1,tf.float32,shape=[shape[3]])
    result,_,_ = tf.nn.fused_batch_norm(inputs,scale,offset)
    offset_m = tf.get_variable(name+'.offset',[n_labels,shape[3]],initializer=tf.initializers.zeros)
    scale_m = tf.get_variable(name+'.scale',[n_labels,shape[3]],initializer=tf.initializers.ones)
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = result*scale[:,None,None,:] + offset[:,None,None,:]
    return result

def nonlinearity(x):
    return tf.nn.relu(x)

def NormalizeG(name, inputs,is_training,labels):
    with tf.variable_scope(name) as scope:
        if NORMALIZATION_G:
            # return batch_norm(inputs, name, is_training=is_training)
            if labels[1] is None:
                return batch_norm(inputs,name,is_training=is_training)
            else:
                return cond_batchnorm(inputs,name,is_training=is_training,labels=labels[0],n_labels=labels[1])
        return inputs

def NormalizeD(name, inputs,labels=None):
    with tf.variable_scope(name) as scope:
        if NORMALIZATION_D:
            if not CONDITIONAL:
                labels = None
            output = tf.transpose(inputs, [0, 3, 1, 2])
            output = lib.ops.layernorm.Layernorm(name, [1, 2, 3], output, labels=labels, n_labels=n_labels)
            return tf.transpose(output, [0, 2, 3, 1])
        return inputs


def linear(input_, output_size, name, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear" + name):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def avg_pool(x,name,is_training,labels):
    x = tf.nn.avg_pool(
        x,
        [1, 3, 3, 1],
        [1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
    )
    return NormalizeG(name+'a', x,is_training,labels)

def max_pool(x,name,is_training,labels):
    x = tf.nn.max_pool(
        x,
        [1, 3, 3, 1],
        [1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
    )
    return NormalizeG(name+'m', x,is_training,labels)


def skip_connect(x):
    return tf.identity(x)


def sep_conv_3x3(x, name,is_training,labels):
    return SepConv(x, [3, 3], [1, 1], name,is_training,labels)


def sep_conv_5x5(x, name,is_training,labels):
    return SepConv(x, [5, 5], [1, 1], name,is_training,labels)


def sep_conv_7x7(x, name,is_training,labels):
    return SepConv(x, [7, 7], name,is_training,labels)


def dil_conv_3x3(x, name,is_training,labels):
    return DilConv(x, [3, 3], [1, 1], 2, name,is_training,labels)


def dil_conv_5x5(x, name,is_training,labels):
    return DilConv(x, [5, 5], [1, 1], 2, name,is_training,labels)


def DilConv(x, kernel_size, stride, rate, name,is_training,labels):
    C_in = x.get_shape()[-1].value
    x = tf.nn.relu(x)
    x = slim.separable_convolution2d(x, C_in, kernel_size, depth_multiplier=1, stride=stride, rate=rate)
    x =  NormalizeG(name, x,is_training,labels)#slim.batch_norm(x)

    return x


def SepConv(x, kernel_size, stride, name,is_training,labels):
    with tf.variable_scope(name):
        x = tf.nn.relu(x)
        C_in = x.get_shape()[-1].value
        x = slim.separable_convolution2d(x, C_in, kernel_size, depth_multiplier=1, stride=stride)
        x = NormalizeG(name+'n1', x,is_training,labels)#slim.batch_norm(x)
        x = slim.separable_convolution2d(x, C_in, kernel_size, depth_multiplier=1)
        x = NormalizeG(name+'n2', x,is_training,labels)#slim.batch_norm(x)
        return x


def conv_3(x,name,is_training,labels):
    with tf.variable_scope(name):
        C_in = x.get_shape()[-1].value
        x = tf.nn.relu(NormalizeG(name, x,is_training,labels))
        conv = tf.layers.conv2d(x, C_in, 3, padding='same', name='conv3')
        return conv


def conv_1(x,  name,is_training,labels):
    with tf.variable_scope(name):
        C_in = x.get_shape()[-1].value
        x = tf.nn.relu(NormalizeG(name, x,is_training,labels))
        conv = tf.layers.conv2d(x, C_in, 1, padding='same', name='conv3')
        return conv


def deconv_6(input_, output_shape,  name,is_training,labels):
    with tf.variable_scope(name):
        deconv = tf.nn.relu( NormalizeG(name, input_,is_training,labels))
        deconv = deconv2d(deconv, output_shape, k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02, name="deconv2d" + name,
                          with_w=False)
        return deconv


def deconv_4(input_, output_shape,  name,is_training,labels):
    with tf.variable_scope(name):
        deconv = tf.nn.relu(NormalizeG(name, input_,is_training,labels))
        deconv = deconv2d(deconv, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="deconv2d" + name,
                          with_w=False)
        return deconv


def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w, stddev, name, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def uc3(inputs, output_shape,  name,is_training,labels):
    output = UpsampleConv(inputs, output_shape, 3,  name,is_training,labels)
    return output


def uc1(inputs, output_shape,  name,is_training,labels):
    output = UpsampleConv(inputs, output_shape, 1,  name,is_training,labels)
    return output


def UpsampleConv(inputs, output_shape, filter_size,  name,is_training,labels):
    with tf.variable_scope(name) as scope:
        shape = inputs.get_shape().as_list()
        stride = int(output_shape[1] / shape[1])
        output = tf.image.resize_nearest_neighbor(images=inputs, size=[shape[1] * stride, shape[2] * stride])
        output = tf.layers.conv2d(output, output_shape[3], filter_size, padding='same', name=name)
        output = tf.nn.relu(NormalizeG(name, output,is_training,labels))
    return output