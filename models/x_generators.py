import tensorflow as tf
# from models.ResidualBlocks import *
from operations_found_models import *


def x_generator1(in_,arch,labels,DIM,is_training,image_size, reuse):
    cur_size = image_size[0]
    while cur_size >= 8 and cur_size % 2 == 0:
        cur_size = int(cur_size/2)
    with tf.variable_scope("x_generator", reuse=reuse) as scope:
        output = tf.layers.dense(in_, cur_size*cur_size*DIM, name='Input')
        output1 = tf.reshape(output, [-1, cur_size, cur_size, DIM])
        BATCH_SIZE = in_.shape.as_list()[0]

        if arch == 'cifar10_n1_resnet_const_end_3e1_no_tg_200':
            s0 = output1
            s1 = sep_conv_5x5(s0,'s1',is_training,labels)
            s2 = uc1(s0, [BATCH_SIZE, cur_size * 2, cur_size * 2, DIM], 's2',is_training,labels) + uc3(s1, [BATCH_SIZE, cur_size * 2, cur_size * 2, DIM], 's2_1',is_training,labels)
            s3 = sep_conv_3x3(s2, 's3',is_training,labels)
            s4 = deconv_4(s2, [BATCH_SIZE, cur_size * 4, cur_size * 4, DIM], 's4',is_training,labels) + uc3(s3, [BATCH_SIZE, cur_size * 4, cur_size * 4, DIM], 's4_1',is_training,labels)
            s5 = conv_3(s4,'s5',is_training,labels)
            s6 = deconv_6(s5, [BATCH_SIZE, cur_size * 8, cur_size * 8, DIM], 's6',is_training,labels)

            output = NormalizeG('OutputN', s6, is_training, labels=labels)
            output = nonlinearity(output)
            output = tf.layers.conv2d(output, image_size[2], 3, padding='same', name='Output')
            output = tf.tanh(output)
            return output


        if arch == 'cifar10_n2_resnet_const_end_3e1_no_tg_200':
            s0 = output1
            s1 = dil_conv_5x5(s0,'s1',is_training,labels)
            s2 = sep_conv_3x3(s0, 's2',is_training,labels) + sep_conv_5x5(s1, 's2_1',is_training,labels)
            s3 = deconv_6(s1, [BATCH_SIZE, cur_size * 2, cur_size * 2, DIM], 's3',is_training,labels) + deconv_4(s2, [BATCH_SIZE, cur_size * 2, cur_size * 2, DIM], 's3_1',is_training,labels)
            s4 = sep_conv_3x3(s3, 's4',is_training,labels)
            s5 = sep_conv_5x5(s3, 's5',is_training,labels) + sep_conv_5x5(s4, 's5_1',is_training,labels)
            s6 = deconv_4(s3, [BATCH_SIZE, cur_size * 4, cur_size * 4, DIM], 's6',is_training,labels) + deconv_4(s5, [BATCH_SIZE, cur_size * 4, cur_size * 4, DIM], 's6_1',is_training,labels)
            s7 = dil_conv_5x5(s6,'s7',is_training,labels)
            s8 = sep_conv_5x5(s7, 's8',is_training,labels)
            s9 = uc1(s6, [BATCH_SIZE, cur_size * 8, cur_size * 8, DIM], 's9',is_training,labels) + uc1(s8, [BATCH_SIZE, cur_size * 8, cur_size * 8, DIM], 's9_1',is_training,labels)

            output = NormalizeG('OutputN', s9, is_training, labels=labels)
            output = nonlinearity(output)
            output = tf.layers.conv2d(output, image_size[2], 3, padding='same', name='Output')
            output = tf.tanh(output)
            return output

        if arch == 'stl10_n1_3e2_200':
            s0 = output1
            s1 = sep_conv_5x5(s0,'s1',is_training,labels)
            s2 = uc3(s1, [BATCH_SIZE, cur_size * 2, cur_size * 2, DIM], 's2',is_training,labels) + uc3(s0, [BATCH_SIZE, cur_size * 2, cur_size * 2, DIM], 's2_1',is_training,labels)
            s3 = avg_pool(s2, 's3',is_training,labels)
            s4 = deconv_4(s3, [BATCH_SIZE, cur_size * 4, cur_size * 4, DIM], 's4',is_training,labels)
            s5 = conv_3(s4,'s5',is_training,labels)
            s6 = uc3(s4, [BATCH_SIZE, cur_size * 8, cur_size * 8, DIM], 's6',is_training,labels) + uc1(s5, [BATCH_SIZE, cur_size * 8, cur_size * 8, DIM], 's6_1',is_training,labels)

            output = NormalizeG('OutputN', s6, is_training, labels=labels)
            output = nonlinearity(output)
            output = tf.layers.conv2d(output, image_size[2], 3, padding='same', name='Output')
            output = tf.tanh(output)
            return output

# def x_generator2(in_,labels,DIM,is_training,image_size, reuse):
#     cur_size = image_size[0]
#     n_layer = -1
#     while cur_size >= 8 and cur_size % 2 == 0:
#         cur_size = int(cur_size/2)
#         n_layer+=1
#     with tf.variable_scope("x_generator", reuse=reuse) as scope:
#         factor = min(2**n_layer,8)
#         output = tf.layers.dense(in_, cur_size*cur_size*DIM*factor, name='Input')
#         output = tf.reshape(output, [-1, cur_size, cur_size, DIM*factor])
#         i=1
#         while cur_size!=image_size[0]:
#             factor = min(2**(n_layer-i+1), 8)
#             output = ResidualBlock_Up('ResidualBlock_Up.'+str(i), DIM*factor, 3,is_training, inputs=output, labels=labels)
#             cur_size = output.get_shape().as_list()[1]
#             i += 1
#         output = NormalizeG('OutputN', output,is_training,labels)
#         output = nonlinearity(output)
#         output = tf.layers.conv2d(output,image_size[2],3, padding='same', name='Output')
#         output = tf.tanh(output)
#         return output