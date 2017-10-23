import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

from nets import *
from datas import *
from gans import *



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    folder = './'
    model = sys.argv[1]
    img_size = sys.argv[2]
    class_id = sys.argv[3]
    class_num = sys.argv[4]
    restore = sys.argv[5]
    print('Model: '+model +'; Img_Resize: '+img_size +'; class_ID: '+class_id +'; class_num: '+class_num +'; Restore_ckpt: '+restore)
    
    if model == 'gan_c' :
        
        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        sample_folder = folder+'Samples/DR_'+img_size+'_'+new_class_id+'_'+'cwgan_conv'
        ckpt_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_class_id+'/'
        restore_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_class_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv_condition(size=int(img_size))
        #classifier = C_conv(size=int(img_size),class_num=int(class_num))
        #classifier = nielsen_net(class_num=int(class_num))
        classifier = net_in_net(class_num=int(class_num))

        min_class_id = class_id.split(',')[0]
        data_all = mydata(size=int(img_size), classes=class_id, class_num=int(class_num))
        data_min = mydata(size=int(img_size), classes=min_class_id, class_num=1)

        # run
        GAN = GAN_Classifier(generator, discriminator, classifier, data_all, data_min)
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = 16, restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = 16, restore=False)

#######
    elif model == 'wgan':
        sample_folder = folder+'Samples/DR_'+img_size+'_'+class_id+'_'+loss_type+'gan_conv'
        ckpt_folder = folder+'ckpt/W_GAN_'+img_size+'_'+class_id+'/'
        restore_folder = folder+'ckpt/W_GAN_'+img_size+'_'+class_id+'/'
        
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

        # param
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv(size=int(img_size))
        
        data = mydata(size=int(img_size), classes=class_id, class_num=int(class_num))
        
        # run
        GAN = WGAN(generator, discriminator, data)
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = 64, restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = 64, restore=False)
        
    elif model == 'c' :

        class_id_pre = class_id.split(',')[0]
        for i in range(1,int(class_num)):
            class_id_post = class_id.split(',')[i]
            #print(class_id_pre,class_id_post)
            class_id_pre = class_id_pre + '-' + class_id_post
        new_class_id = class_id_pre
        print(new_class_id)

        ckpt_folder = folder+'ckpt/classifier_'+img_size+'_'+new_class_id+'/'
        restore_folder = folder+'ckpt/classifier_'+img_size+'_'+new_class_id+'/'

        # param
        #classifier = nielsen_net(class_num=int(class_num))
        classifier = net_in_net(class_num=int(class_num))
         
        data = mydata(size=int(img_size), classes=class_id, class_num=int(class_num))

        # run
        C = Classifer(classifier, data)
        if restore == 'True':
            C.restore_ckpt(restore_folder)
            C.train(ckpt_dir=ckpt_folder, restore=True)
        else:
            C.train(ckpt_dir=ckpt_folder, restore=False)


    elif model == 'began' :
        sample_folder = folder+'Samples/DR_'+img_size+'_'+class_id+'_BEgan_conv'
        ckpt_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+class_id+'/'
        restore_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+class_id+'/'

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        # param
        generator = G_conv_BEGAN(size=int(img_size))
        discriminator_tmp = D_conv(size=int(img_size))
        discriminator = D_conv_BEGAN(size=int(img_size))
        
         
        data = mydata(size=int(img_size), classes=class_id, class_num=int(class_num))

        # run
        GAN = BEGAN(generator, discriminator, data, flag=False)
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            #GAN.restore_ckpt(folder+'ckpt/W_GAN_64_13/')
            #GAN.discriminator = discriminator
            #GAN.sess.run(tf.variables_initializer(GAN.discriminator.vars))
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, restore=False)

    else:
        print('Wrong model')
