from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
import scipy
import re
import pdb
from .nnlib import *
from .parameters import arch_para, hparams, Parameters
from util import read_image, comp_confusionmat
import tensorflow.contrib.slim as slim

from .tensorflow_vgg import custom_vgg19
from .layer_modules import prog_ch, tf_MILloss_xentropy, tf_loss_xentropy, tf_MILloss_accuracy, tf_background, syntax_loss, tf_accuracy, create_canonical_coordinates, oper_random_geo_perturb, oper_img2img, style_layer_loss, tf_frequency_weight, refinernet, infernet 


def conv2d(input_, output_dim, ks=3, s=2, stddev=0.02, padding='VALID', name="conv2d"):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        padsz = math.ceil((ks - s) * 0.5)
        if padsz != 0:
            input_ = tf.pad(input_,
                        tf.constant([[0, 0], [padsz, padsz], [padsz, padsz], [0, 0]]),
        mode='SYMMETRIC')
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                            activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev)
                            # biases_initializer=None
                            )
def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME',                            activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev)
                            # biases_initializer=None
                            )

def discriminator(image, params = dict(), name="discriminator"):

    feat_ch = int(params.get('feat_ch', 64))
    noise_sigma = params.get('noise_sigma', 3./255.)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        image = image + tf.random_normal(tf.shape(image), stddev = noise_sigma)

        h0 = lrelu(conv2d(image, feat_ch, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim) 80
        h1 = lrelu(conv2d(h0, feat_ch*2, name='d_h1_conv'))
        # h1 is (64 x 64 x self.df_dim*2) 40
        h2 = lrelu(conv2d(h1, feat_ch*4, name='d_h2_conv'))
        # h2 is (32x 32 x self.df_dim*4) 20
        h3 = lrelu(conv2d(h2, feat_ch*8, name='d_h3_conv'))
        # h3 is (32 x 32 x self.df_dim*8) 10
        h4 = lrelu(conv2d(h3, feat_ch*8, s=1, name='d_h4_conv'))
        h5 = conv2d(h4, 1, s=1, name='d_h4_pred')
        # h4 is (32 x 32 x 1)

        return h5


def discriminator_cond(image, instruction, params = dict(), name="discriminator"):

    feat_ch = int(params.get('feat_ch', 64))
    noise_sigma = params.get('noise_sigma', 3./255.)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is 256 x 256 x input_c_dim
        image = image + tf.random_normal(tf.shape(image), stddev = noise_sigma)

        t_onehot = tf.one_hot(tf.squeeze(instruction),depth=prog_ch,dtype=tf.float32)
        t_embed_inst = conv2d(t_onehot, 4, ks=1, s=1)
        t_embed_inst = tf.image.resize_images(t_embed_inst, 
                                            [image.get_shape()[1], image.get_shape()[2]],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )

        h0 = lrelu(conv2d(tf.concat([image, t_embed_inst], axis=-1), feat_ch, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim) 80
        h1 = lrelu(conv2d(h0, feat_ch*2, name='d_h1_conv'))
        # h1 is (64 x 64 x self.df_dim*2) 40
        h2 = lrelu(conv2d(h1, feat_ch*4, name='d_h2_conv'))
        # h2 is (32x 32 x self.df_dim*4) 20
        h3 = lrelu(conv2d(h2, feat_ch*8, s=1, name='d_h3_conv'))
        # h3 is (32 x 32 x self.df_dim*8) 10
        # h4 = lrelu(conv2d(h3, feat_ch*8, s=1, name='d_h4_conv'))
        h4 = conv2d(h3, 1, s=1, name='d_h4_pred')
        # h4 is (32 x 32 x 1)

        return h4






def model_composited_RFI_complexnet(t_imgs_dict, t_labels_dict, params = dict()):
    '''
    Compose the full network model
    '''
    net = Parameters()
    net.inputs = t_imgs_dict
    net.imgs = dict()
    net.resi_imgs = dict()  # rend | tran | real
    net.resi_imgs_noaug = dict()  # rend | tran | real
    net.latent = dict()     # rend | tran | real
    net.logits = dict()     # rend | tran | real
    net.instr  = dict()     # rend | tran | real
    net.resi_outs = dict()  # rend | tran | real
    net.activations = dict()
    is_train = params['is_train']
    
    

    # activations
    def store_act(name, target, activations):#用于存储网络中间层的激活值，以便进行模型调试。（名称，激活目标，激活值）
        if name not in net.activations:
            net.activations[name] = dict()
        net.activations[name][target] = activations

    # input augmentation
    coords_res = int(params.get('coords_res', 20))
    batch_size = net.inputs['real'].get_shape()[0]
    t_canonical_coords, blk_size = create_canonical_coordinates(batch_size, 160, coords_res)#blk_size =8
    coords_sigma = params.get('coords_sigma', 1.0) * blk_size * 0.2
    for key, t_img in net.inputs.items():
        net.resi_imgs_noaug[key] = t_img#此时net.resi_imgs_noaug[key]为原始img
        
        # for the RFI net
        if is_train and key.startswith('real') and params.get('local_warping', 0):
            # local warp augmentation
            with tf.variable_scope("input"):
                net.imgs[key], _, __ = oper_random_geo_perturb(t_img, t_canonical_coords, coords_sigma)
        else:
            net.imgs[key] = t_img # no augmentation  ##此时net.imgs[key]为原始img

    # mean inputs and residuals
    net.mean_imgs = dict()
    for key, t_img in net.imgs.items(): #遍历net.imgs字典中的所有图像，其中每个键表示一种图像类型（例如'real'、'rend'、'tran'），每个值是相应的 TensorFlow 张量。
        value = params.get('mean_' + key, 0.5)   #使用键'mean_' + key从params字典中获取均值。
        if isinstance(value, str):  #  如果这个均值以字符串形式提供，它会假设它是一个图像文件的路径，读取图像并扩展其维度，使其与 TensorFlow 张量兼容。
            print('isinstance==str')
            value = read_image(value)
            value = np.expand_dims(value, axis=0)
            print('mean image', value.shape)
        net.mean_imgs[key] = value         #0.5
        net.resi_imgs[key] = t_img - value #原始img-0.5
        
 
        
        

        if is_train: # if training
            noise_sigma = params.get('noise_sigma', 3./255.)#从参数中获取了噪声的标准差（默认为3/255）
            t_noise = tf.random_normal(tf.shape(t_img), stddev = noise_sigma)#然后使用 tf.random_normal 函数生成一个与输入图像相同形状的张量，其中的值是服从均值为0、标准差为 noise_sigma 的正态分布的随机数。
            net.resi_imgs[key] = net.resi_imgs[key] + t_noise   #向残差图像添加了一些噪声，以增加数据的多样性，从而提高模型的鲁棒性和泛化能力。
        net.resi_imgs_noaug[key] = net.resi_imgs_noaug[key] - value#这行代码将原始图像减去了其均值，得到了残差图像，然后将其存储在net.resi_imgs_noaug[key]中

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', t_imgs_dict.keys())

    # create generator
    with tf.variable_scope("generator"):

        def transformer(t_input, name):
            with runits('in_relu') as activations:
                t_gene_img = refinernet(t_input, 1, params=params, name='transformer_r2s')
                t_gene_img = tf.nn.tanh(t_gene_img)*0.5        #对生成的图像进行了 tanh 激活，并将其缩放到 [-0.5, 0.5] 的范围内
                # 160x160
                net.resi_outs[name] = t_gene_img                 #将生成的图像保存到网络的 resi_outs 字典中，键为名称 name。
                store_act(name, 'real2syn', activations)
            return t_gene_img

        def encoder(t_input, name):
            with runits('relu') as activations:
                t_logits = infernet(t_input, params=params, name='img2prog')
                t_instr = tf.argmax(t_logits, axis=3, name="prediction")
                net.latent[name] = t_logits
                net.logits[name] = t_logits
                net.instr[name]  = tf.expand_dims(t_instr, axis=3)
                store_act(name, 'img2prog', activations)
            return t_logits

        # program synthesis (encoding)
        curdataname = 'real'
        fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
        _ = encoder(fakesyn_img, curdataname)
        
        if is_train:
            # program synthesis (encoding)
            curdataname = 'rend'
            _ = encoder(net.resi_imgs[curdataname], curdataname)

            # CH0118_14
            curdataname = 'tran'
            if curdataname in net.resi_imgs.keys():
                fakesyn_img = transformer(net.resi_imgs[curdataname], curdataname)
                _ = encoder(fakesyn_img, curdataname)

    return net


# #refinernet
def refinernet(t_img, out_ch, params, name='img2img'):
    '''
    Translate image domain with resnet 160x160->160x160
    '''
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"  )
    feat_ch = int(params.get('feat_ch', 64))
    rblk_num = int(params.get('rblk_num', 6))
    conv_type = params.get('conv_type', 'conv_pad')
    if conv_type == 'conv_pad':
        conv_fn = conv_pad
  
    else:
        raise ValueError('Unsupported convolution type %s' % conv_type)

    rblk = [resi, [[conv_fn, feat_ch], [runit], [conv_fn, feat_ch]]]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#encoder
        t_down1 = NN('t_down1',
            [t_img, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        t_res1 = NN('t_res1',
            [t_down1, *[rblk for _ in range(7)]])
        print("res1=%s" % t_res1)
        t_norm1= NN('t_norm1',
            [t_res1,
            [conv_fn,feat_ch,1,1],[runit, 'in_relu']
            ])
        
        t_down2 = NN('t_down2',
            [t_res1, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],  #128          
            ])
        print("t_down2 =%s" % t_down2 )
        t_res2 = NN('t_res2',
            [t_down2, *[rblk for _ in range(6)]])
        print("res2=%s" % t_res2)
        t_norm2= NN('norm2',
            [t_res2,
            [conv_fn,feat_ch,1,1],[runit, 'in_relu']
            ])
        t_up2= NN('t_up2',
            [t_norm2,
            [upsample],
            [conv_fn, feat_ch], [runit]
            ])
        
        
        
        t_down3 = NN('t_down3',
            [t_res2, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],#256
            ])
        t_res3 = NN('t_res3',
            [t_down3, *[rblk for _ in range(2)]])
        t_norm3= NN('norm3',
            [t_res3,
            [conv_fn,feat_ch,1,1],[runit, 'in_relu']
            ])
        t_up3= NN('t_up3',
            [t_norm3,
            [upsample],
            [conv_fn, feat_ch], [runit],
            [upsample],
            [conv_fn, feat_ch], [runit],             
            ])


#         t_out  = NN('resnet3',
#             [tf.concat([t_norm1, t_up2,t_up3], -1),
#                 [conv_fn,feat_ch,1,1],[runit, 'in_relu'],
#                 [upsample],
#                 [conv_fn, feat_ch], [runit],#上采样后卷积可以平滑特征图
#                 [upsample],
#                 [conv_fn, feat_ch], [runit],
#                 [conv_fn, out_ch, 1, 1]
#             ])
#         print("t_out=%s" % t_out)
#     return t_out








#infernet
def infernet(t_img, params = dict(), name='img2prog'):
    '''
    Translate image domain with resnet 160x160->20x20
    '''
    feat_ch = int(params.get('feat_ch', 64))
    rblk_num = int(params.get('rblk_num', 6))
    conv_type = params.get('conv_type', 'conv_pad')
    if conv_type == 'conv_pad':
        conv_fn = conv_pad
    elif conv_type == 'conv':
        conv_fn = conv
    else:
        raise ValueError('Unsupported convolution type %s' % conv_type)

    rblk = [resi, [[conv_fn, feat_ch], [runit], [conv_fn, feat_ch]]]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # 第一次卷积，输出尺寸为 80x80
        t_act1_80 = NN('img2feat_80',
            [t_img, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        print('t_act1_80 shape:', t_act1_80.shape)  # 输出：(batch_size, 80, 80, feat_ch)
        
        # 第二次卷积，输出尺寸为 40x40
        t_act1_40 = NN('img2feat_40',
            [t_act1_80, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        print('t_act1_40 shape:', t_act1_40.shape)  # 输出：(batch_size, 40, 40, feat_ch)
        
        # 第三次卷积，输出尺寸为 20x20
        t_act1 = NN('img2feat_20',
            [t_act1_40, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            ])
        print('t_act1 shape:', t_act1.shape)  # 输出：(batch_size, 20, 20, feat_ch)
        
        # feat2feat处理
        t_act2_80 = NN('feat2feat_80',
            [t_act1_80, *[rblk for _ in range(rblk_num)]])
        print('t_act2_80 shape:', t_act2_80.shape)  # 输出：(batch_size, 80, 80, feat_ch)
        
        t_act2_40 = NN('feat2feat_40',
            [t_act1_40, *[rblk for _ in range(rblk_num)]])
        print('t_act2_40 shape:', t_act2_40.shape)  # 输出：(batch_size, 40, 40, feat_ch)
        
        t_act2 = NN('feat2feat_20',
            [t_act1, *[rblk for _ in range(rblk_num)]])
        print('t_act2 shape:', t_act2.shape)  # 输出：(batch_size, 20, 20, feat_ch)
      
        t_act2_80 = NN('img2feat-80',
            [t_act2_80, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            
            ])
        

        t_act2_40 = NN('img2feat-40',
            [t_act2_40, 
            [conv_fn, feat_ch, 2], [runit, 'in_relu'],
            
            
            ])
            
    
        # 将三个尺度的特征进行concat融合
        t_act_concat = tf.concat([t_act2_80, t_act2_40, t_act2], axis=-1)
        print('t_act_concat shape:', t_act_concat.shape)  # 输出：(batch_size, 20, 20, 3 * feat_ch)
        
        # 最终的输出
        t_out  = NN('feat2prog',
            [t_act_concat,
                [conv_fn, feat_ch], [runit],
                [conv_fn, prog_ch, 1, 1]
            ])
        print('t_out shape:', t_out.shape)  # 输出：(batch_size, 20, 20, prog_ch)
    return t_out

##############################################################################





def total_loss_RFI(net, t_inst_dict, params = dict()):
    loss_dict_Disc = dict()
    loss_dict_Gene = dict()
    metrics = dict()

    # replay switch
    replay_worst = params.get('replay_worst', 0)

    # extract instructions
    t_inst_real = t_inst_dict['instr_real']
    if 'instr_synt' in t_inst_dict.keys():
        t_inst_synt = t_inst_dict['instr_synt']
    if replay_worst:
        t_inst_wors = t_inst_dict['worst']

    # get dimensions
    batch_size, h, w, _ = net.imgs['real'].get_shape()

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup' and name != 'worst', net.imgs.keys())


    # replay switch
    replay_worst = params.get('replay_worst', 0)

    # instruction masking and weighting
    ones = tf.ones_like(t_inst_synt, tf.float32)
    zeros = tf.zeros_like(t_inst_synt, tf.float32)
    bg_type = params.get('bg_type', 'global')
    t_bg_synt = tf_background(t_inst_synt, bg_type)
    t_bg_real = tf_background(t_inst_real, bg_type)
    if replay_worst:
        t_bg_wors = tf_background(t_inst_wors, bg_type)
    t_synt_mask = tf.where(t_bg_synt, zeros, ones)
    t_real_mask = tf.where(t_bg_real, zeros, ones)
    if replay_worst:
        t_wors_mask = tf.where(t_bg_wors, zeros, ones)
    bg_weight = params.get('bg_weight', 0.1)
    if isinstance(bg_weight, str):
        masked = bg_weight.startswith('mask_')
        if masked:
            bg_weight = bg_weight[5:]
        t_synt_weight = tf_frequency_weight(t_inst_synt, bg_weight)
        t_real_weight = tf_frequency_weight(t_inst_real, bg_weight)
        if replay_worst:
            t_wors_weight = tf_frequency_weight(t_inst_wors, bg_weight)
        if masked:
            t_synt_weight = tf.where(t_bg_synt, 0.1 * t_synt_weight, t_synt_weight)
            t_real_weight = tf.where(t_bg_real, 0.1 * t_real_weight, t_real_weight)
            if replay_worst:
                t_wors_weight = tf.where(t_bg_wors, 0.1 * t_wors_weight, t_wors_weight)
    else:
        t_synt_weight = tf.where(t_bg_synt, bg_weight * ones, ones)
        t_real_weight = tf.where(t_bg_real, bg_weight * ones, ones)
        if replay_worst:
            t_wors_weight = tf.where(t_bg_wors, bg_weight * ones, ones)
    t_simg_weight = tf.image.resize_bilinear(t_synt_weight, [h, w])
    t_rimg_weight = tf.image.resize_bilinear(t_real_weight, [h, w])
    if replay_worst:
        t_wimg_weight = tf.image.resize_bilinear(t_wors_weight, [h, w])

    # store background for debugging
    net.bg = dict()
    net.bg['synt'] = t_bg_synt
    net.bg['real'] = t_bg_real
    if replay_worst:
        net.bg['worst'] = t_bg_wors
    
    # create discriminator networks if needed for loss
    net.discr = {
        # 'instr': dict(), 
        # 'latent': dict(), 
        'image': dict(),
        #'wgan_grad': dict()
    }

     # summon VGG19
    if params.get('bvggloss', 0):
        if params.get('vgg16or19', '16') == '16':
            net.vggobj = custom_vgg19.Vgg16()
        else:
            net.vggobj = custom_vgg19.Vgg19()
        net.vgg = dict()

        # GT synthetic
        curdataname = 'rend'
        net.vgg['gt_' + curdataname] = net.vggobj.build(net.resi_imgs[curdataname])
        curdataname = 'real'
        net.vgg['gt_' + curdataname] = net.vggobj.build(net.resi_imgs[curdataname])
        # generated data
        curdataname = 'real'
        net.vgg[curdataname] = net.vggobj.build(net.resi_outs[curdataname])
        curdataname = 'tran'
        net.vgg[curdataname] = net.vggobj.build(net.resi_outs[curdataname])

    if params.get('discr_img', 0):
        with tf.variable_scope("discriminator"):
            # GT synthetic
            curdataname = 'rend'
            t_domain = discriminator_cond(net.resi_imgs[curdataname], 
                                            t_inst_synt,
                                            params, name="image_domain")
            net.discr['image']['gt_' + curdataname] = t_domain

            curdataname = 'real'
            t_domain = discriminator_cond(net.resi_imgs[curdataname], 
                                            t_inst_real,
                                            params, name="image_domain")
            net.discr['image']['gt_' + curdataname] = t_domain

            # generated data
            curdataname = 'tran'
            t_domain = discriminator_cond(net.resi_outs[curdataname], 
                                            t_inst_synt,
                                            params, name="image_domain")
            net.discr['image'][curdataname] = t_domain

            curdataname = 'real'
            t_domain = discriminator_cond(net.resi_outs[curdataname], 
                                            t_inst_real,
                                            params, name="image_domain")
            net.discr['image'][curdataname] = t_domain


    # generator and discriminator losses
    with tf.variable_scope("loss"):

        # adversarial loss for image
        discr_type = params.get('discr_type', 'l2')
        name = 'gt_rend'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis
        name = 'gt_real'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis/3.
        name = 'tran'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis/3.
        name = 'real'
        t_discr = net.discr['image'][name]
        loss_dis = tf_loss_with_select(t_discr, -tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Disc['loss_D_image/' + name] = loss_dis/3.

        name = 'tran'
        t_discr = net.discr['image'][name]
        loss_gen = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Gene['loss_G_image/' + name] = loss_gen 

        name = 'real'
        t_discr = net.discr['image'][name]
        loss_gen = tf_loss_with_select(t_discr, tf.ones_like(t_discr), which_loss = discr_type)
        loss_dict_Gene['loss_G_image/' + name] = loss_gen 
        

        def fn_downsize(images):
            smoother = Smoother({'data':images}, 11, 2.)
            images = smoother.get_output()
            return tf.image.resize_bilinear(images, [20,20])
        
        if params.get('discr_img', 1):
            curdataname = 'real'
            ae_loss_type = params.get('ae_loss_type', 'smooth_l1')
            loss_unsup = tf_loss_with_select(
                                fn_downsize(
                                    net.resi_imgs[curdataname]
                                ), 
                                fn_downsize(
                                    net.resi_outs[curdataname]
                                ),
                            which_loss = ae_loss_type)
            # loss_dict_Gene['loss_unsup/rough_real_vs_freal'] = loss_unsup

            loss_unsup = tf_loss_with_select(
                                    net.resi_imgs['rend'], 
                                    net.resi_outs['tran'],
                            which_loss = ae_loss_type)
            # loss_unsup = tf_loss_with_select(net.activations['unsup']['img2prog'][2],
            #                                  net.activations['unsup_feedback']['img2prog'][2],
            #                                  which_loss='smooth_l1')
            loss_dict_Gene['loss_unsup/rough_gtrend_vs_ftrans'] = 100.*loss_unsup

        # VGG perceptual loss
        if params.get('bvggloss', 0):
            curlayer = 'pool3'
            loss_perc_pool2 = 1.*tf_loss_with_select(
                                (1./128.)*net.vgg['gt_real'][curlayer], 
                                (1./128.)*net.vgg['real'][curlayer], 
                                which_loss = 'l2')
            loss_dict_Gene['loss_vgg_percept/' + curlayer] = loss_perc_pool2*0.25 
            # normalize by the number of combinations (real, unsuper, conv2_2, pool3)

            curlayer = 'pool3'
            loss_perc_pool2 = 1.*tf_loss_with_select(
                                (1./128.)*net.vgg['gt_rend'][curlayer], 
                                (1./128.)*net.vgg['tran'][curlayer], 
                                which_loss = 'l2')
            loss_dict_Gene['loss_vgg_percept/' + curlayer] = loss_perc_pool2*0.25 

            # VGG style losses 
            # applied to {synt, real} except {unsup}
            
            lst_lweight = [0.3, 0.5, 1.]
            lst_layers = ['conv1_2', 'conv2_2', 'conv3_3']
            no_gram_layers = float(len(lst_layers))
            for gram_layer, gram_weight in zip(lst_layers, lst_lweight):
                loss_prefix = 'loss_vgg_percept/' + 'gram_' + gram_layer + 'real'

                lst_gts = ['rend',] #['real', 'unsup']
                no_gts = float(len(lst_gts))
                for gts in lst_gts:
                    t_real = net.vgg['gt_rend'][gram_layer]/128.
                    t_synt = net.vgg['real'][gram_layer]/128.
                    t_loss = style_layer_loss(t_real, t_synt, params.get('gram_power', 2))
                    loss_dict_Gene[loss_prefix + 'real2rend'] = gram_weight*t_loss/(no_gram_layers*no_gts)

        # instruction x-entropy
        # applied to {*real*} including {rend_real, tran_real, real_real, real, real_feedback...}
        nsynthetic = len(net.logits.keys()) - 1.
        for name, t_logits in net.logits.items():
            if name.startswith('unsup'): # name.endswith('_real') or 
                continue # adapter network doesn't use entropy

            if name.startswith('real'):
                # name == 'real' or name == 'real_feedback' or name == 'real_gen' or name == 'real_gen_feedback':
                t_instr  = t_inst_real
                t_weight = t_real_weight
            else:
                t_instr  = t_inst_synt
                t_weight = t_synt_weight

            if params.get('bMILloss', 1) and name.startswith('real'):
                loss_xentropy = tf_MILloss_xentropy(labels = tf.squeeze(t_instr),
                                                    logits = t_logits,
                                                    weight = t_weight)
                                    #                      + \
                                    # tf_loss_xentropy(
                                    #                 labels = tf.squeeze(t_instr),
                                    #                 logits = t_logits,
                                    #                 weight = t_weight)[:,1:-1,1:-1,tf.newaxis]*0.5
                loss_xentropy *= (1.-params.get('mix_alpha', 0.5))*3.
            else:
                loss_xentropy = tf_loss_xentropy(
                                            labels = tf.squeeze(t_instr),
                                            logits = t_logits,
                                            weight = t_weight)
                loss_xentropy *= params.get('mix_alpha', 0.5)*3./float(nsynthetic)

            if 'feedback' in name:
                loss_prefix = 'loss_feedback/'
            else:
                loss_prefix = 'loss_xentropy/'
            loss_dict_Gene[loss_prefix + name] = loss_xentropy

        # syntax loss
        # applied to {all} except {unsup}
        syntax_binary = params.get('syntax_binary', 0)
        for name in net.instr.keys():
            if name.startswith('unsup') and not params.get('unsup_syntax', 0): # skip
                continue

            if syntax_binary:
                t_instr = net.instr[name]
            else:
                t_instr = net.logits[name]
            loss_syn = syntax_loss(t_instr, params, syntax_binary)
            loss_dict_Gene['loss_syntax/' + name] = loss_syn

        # accuracy measurements
        net.acc = { 'full' : dict(), 'fg': dict() }
        # applied to {all} except {unsup}
        for name, t_instr in net.instr.items():
            if name.startswith('unsup'):
                continue

            if name == 'real':
                t_label = t_inst_real
                t_mask  = t_real_mask
            else:
                t_label = t_inst_synt
                t_mask  = t_synt_mask

            if params.get('bMILloss', 1):
                # full accuracy (includes bg)
                metrics['accuracy/' + name], acc_batch    = tf_MILloss_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name], fac_batch = tf_MILloss_accuracy(t_label, t_instr, t_mask)

                # storing batch information for worst sample mining
                net.acc['full'][name] = acc_batch
                net.acc['fg'][name]   = fac_batch
            else:
                # full accuracy (includes bg)
                metrics['accuracy/' + name] = tf_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name] = tf_accuracy(t_label, t_instr, t_mask)

            metrics['confusionmat/' + name] = comp_confusionmat(t_instr, 
                                                                t_label, 
                                                                num_classes = prog_ch, 
                                                                normalized_row = True,
                                                                name=name)
    return loss_dict_Disc, loss_dict_Gene, metrics





def total_loss(net, t_inst_synt, t_inst_real, params = dict()):
    loss_dict_Disc = dict()
    loss_dict_Gene = dict()
    metrics = dict()

    # get dimensions
    batch_size, h, w, _ = net.imgs['real'].get_shape()

    # fake targets
    fakes = filter(lambda name: name != 'real' and name != 'unsup', net.imgs.keys())

    # instruction masking and weighting
    ones = tf.ones_like(t_inst_synt, tf.float32)
    zeros = tf.zeros_like(t_inst_synt, tf.float32)
    bg_type = params.get('bg_type', 'global')
    t_bg_synt = tf_background(t_inst_synt, bg_type)
    t_bg_real = tf_background(t_inst_real, bg_type)
    t_synt_mask = tf.where(t_bg_synt, zeros, ones)
    t_real_mask = tf.where(t_bg_real, zeros, ones)
    bg_weight = params.get('bg_weight', 0.1)
    if isinstance(bg_weight, str):
        masked = bg_weight.startswith('mask_')
        if masked:
            bg_weight = bg_weight[5:]
        t_synt_weight = tf_frequency_weight(t_inst_synt, bg_weight)
        t_real_weight = tf_frequency_weight(t_inst_real, bg_weight)
        if masked:
            t_synt_weight = tf.where(t_bg_synt, 0.1 * t_synt_weight, t_synt_weight)
            t_real_weight = tf.where(t_bg_real, 0.1 * t_real_weight, t_real_weight)
    else:
        t_synt_weight = tf.where(t_bg_synt, bg_weight * ones, ones)
        t_real_weight = tf.where(t_bg_real, bg_weight * ones, ones)
    t_simg_weight = tf.image.resize_bilinear(t_synt_weight, [h, w])
    t_rimg_weight = tf.image.resize_bilinear(t_real_weight, [h, w])


    # store background for debugging
    net.bg = dict()
    net.bg['synt'] = t_bg_synt
    net.bg['real'] = t_bg_real

    # create discriminator networks if needed for loss
    net.discr = {
        'instr': dict(), 'latent': dict(), 'image': dict(),
        #'wgan_grad': dict()
    }
    net.resi_aug_wgan = dict()

    # generator and discriminator losses
    with tf.variable_scope("loss"):
        
        # instruction x-entropy
        # applied to {*real*} including {rend_real, tran_real, real_real, real, real_feedback...}
        for name, t_logits in net.logits.items():
            if name.startswith('unsup'): # name.endswith('_real') or 
                continue # adapter network doesn't use entropy

            if (re.search('real', name) is not None) or (not params.get('adapter',0)):
                if name.startswith('real'):
                    # name == 'real' or name == 'real_feedback' or name == 'real_gen' or name == 'real_gen_feedback':
                    t_instr  = t_inst_real
                    t_weight = t_real_weight
                else:
                    t_instr  = t_inst_synt
                    t_weight = t_synt_weight

                if params.get('bMILloss', 1) and name.startswith('real'):
                    loss_xentropy = tf_MILloss_xentropy(labels = tf.squeeze(t_instr),
                                                        logits = t_logits,
                                                        weight = t_weight)
                                    #                      + \
                                    # tf_loss_xentropy(
                                    #                 labels = tf.squeeze(t_instr),
                                    #                 logits = t_logits,
                                    #                 weight = t_weight)[:,1:-1,1:-1,tf.newaxis]*0.5
                else:
                    loss_xentropy = tf_loss_xentropy(
                                                labels = tf.squeeze(t_instr),
                                                logits = t_logits,
                                                weight = t_weight)

                if 'feedback' in name:
                    loss_prefix = 'loss_feedback/'
                else:
                    loss_prefix = 'loss_xentropy/'
                loss_dict_Gene[loss_prefix + name] = loss_xentropy

        # syntax loss
        # applied to {all} except {unsup}
        syntax_binary = params.get('syntax_binary', 0)
        for name in net.instr.keys():
            if name.startswith('unsup') and not params.get('unsup_syntax', 0): # skip
                continue

            if syntax_binary:
                t_instr = net.instr[name]
            else:
                t_instr = net.logits[name]
            loss_syn = syntax_loss(t_instr, params, syntax_binary)
            loss_dict_Gene['loss_syntax/' + name] = loss_syn

        # accuracy measurements
        net.acc = { 'full' : dict(), 'fg': dict() }
        # applied to {all} except {unsup}
        for name, t_instr in net.instr.items():
            if name.startswith('unsup'):
                continue
                
            if name == 'real':
                t_label = t_inst_real
                t_mask  = t_real_mask
            else:
                t_label = t_inst_synt
                t_mask  = t_synt_mask

            if params.get('bMILloss', 1):
                # full accuracy (includes bg)
                metrics['accuracy/' + name], acc_batch    = tf_MILloss_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name], fac_batch = tf_MILloss_accuracy(t_label, t_instr, t_mask)

                # storing batch information for worst sample mining
                net.acc['full'][name] = acc_batch
                net.acc['fg'][name]   = fac_batch
            else:
                # full accuracy (includes bg)
                metrics['accuracy/' + name] = tf_accuracy(t_label, t_instr)
                # fg accuracy
                metrics['accuracy_fg/' + name] = tf_accuracy(t_label, t_instr, t_mask)
            
            metrics['confusionmat/' + name] = comp_confusionmat(t_instr, 
                                                                t_label, 
                                                                num_classes = prog_ch, 
                                                                normalized_row = True,
                                                                name=name)


    return loss_dict_Disc, loss_dict_Gene, metrics
