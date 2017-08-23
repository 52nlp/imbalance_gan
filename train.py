#train.py

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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    folder = '/data/projects/yhliuzzb/'
    model = sys.argv[1]
    img_size = sys.argv[2]
    batch = sys.argv[3]
    defect_id = sys.argv[4]
    defect_num = sys.argv[5]
    restore = sys.argv[6]
    print('------------------------------------------------------------------------------------------')
    print('Model: '+model +'; Img_Resize: '+img_size +'; Batch_Size: '+batch +'; Defect_ID: '+defect_id +'; Defect_num: '+defect_num +'; Restore_ckpt: '+restore)
    print('------------------------------------------------------------------------------------------')
    if model == 'cwgan_adc' :
        
        defect_id_pre = defect_id.split(',')[0]
        for i in range(1,int(defect_num)):
            defect_id_post = defect_id.split(',')[i]
            #print(defect_id_pre,defect_id_post)
            defect_id_pre = defect_id_pre + '-' + defect_id_post
        new_defect_id = defect_id_pre
        print(new_defect_id)
        
        sample_folder = folder+'Samples/adc_'+img_size+'_'+new_defect_id+'_'+'cwgan_conv'
        ckpt_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_defect_id+'/'
        restore_folder = folder+'ckpt/CW_GAN_'+img_size+'_'+new_defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

# param
generator = G_conv(size=int(img_size),is_tanh=True)
    discriminator = D_conv_condition(size=int(img_size))
        classifier = C_conv(size=int(img_size),class_num=int(defect_num))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))
        
        # run
        GAN = CGAN_Classifier(generator, discriminator, classifier, data, loss_type = 'W')
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=True)
        else:
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=False)

elif model == 'clsgan_c' :
    sample_folder = './Samples/mnist_clsgan_conv'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

    # param
    generator = G_conv_mnist(is_tanh=True)
        discriminator = D_conv_mnist()
        classifier = C_conv_mnist()
        
        data = mnist(is_tanh=True)
        
        # run
        clsgan_conv = CGAN_Classifier(generator, discriminator, classifier, data, loss_type = 'LS')
        clsgan_conv.train(sample_folder)
#######
elif model == 'wgan_adc' :
    sample_folder = folder+'Samples/adc_'+img_size+'_'+defect_id+'_'+'wgan_conv'
        ckpt_folder = folder+'ckpt/'+'W_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/'+'W_GAN_'+img_size+'_'+defect_id+'/'
        
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

    # param
    generator = G_conv(size=int(img_size),is_tanh=True)
        discriminator = D_conv(size=int(img_size))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))
        
        # run
        GAN = WorLS_GAN(generator, discriminator, data)
        if restore == 'True':
            GAN.restore_ckpt(restore_folder)
            GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=True)
else:
    GAN.train(sample_folder, ckpt_dir=ckpt_folder, batch_size = int(batch), restore=False)
    
    elif model == 'c_adc' :
        
        defect_id_pre = defect_id.split(',')[0]
        for i in range(1,int(defect_num)):
            defect_id_post = defect_id.split(',')[i]
            #print(defect_id_pre,defect_id_post)
            defect_id_pre = defect_id_pre + '-' + defect_id_post
        new_defect_id = defect_id_pre
        print(new_defect_id)
        
        sample_folder = folder+'Samples/adc_'+img_size+'_'+new_defect_id+'_'+'classifier'
        ckpt_folder = folder+'ckpt/classifier_'+img_size+'_'+new_defect_id+'/'
        restore_folder = folder+'ckpt/classifier_'+img_size+'_'+new_defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

# param
classifier = C_conv(size=int(img_size),class_num=int(defect_num))
    
    data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))
        
        # run
        c = Classifier(classifier, data)
        if restore == 'True':
            c.restore_ckpt(restore_folder)
            c.train_classifier(sample_folder, ckpt_dir=ckpt_folder, restore=True)
        else:
            c.train_classifier(sample_folder, ckpt_dir=ckpt_folder, restore=False)


elif model == 'began_adc' :
    sample_folder = folder+'Samples/adc_'+img_size+'_'+defect_id+'_BEgan_conv'
        ckpt_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/BE_GAN_'+img_size+'_'+defect_id+'/'
        
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        # param
        generator = G_conv_BEGAN(size=int(img_size))
        discriminator_tmp = D_conv(size=int(img_size))
        discriminator = D_conv_BEGAN(size=int(img_size))
        
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))
        
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

elif model == 'began' :
    sample_folder = 'Samples/mnist_began_conv'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

    # param
    generator = G_conv_mnist(is_tanh=True)
        discriminator = D_conv_mnist_BEGAN()
        
        
        data = mnist(is_tanh=True)
        
        # run
        began_conv = BEGAN(generator, discriminator, data)
        
        began_conv.train(sample_folder)

elif model == 'cbegan_c' :
    sample_folder = 'Samples/mnist_cbegan_conv'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)

    # param
    generator = G_conv_mnist(is_tanh=True)
        discriminator = D_conv_mnist_BEGAN()
        classifier = C_conv_mnist()
        
        data = mnist(is_tanh=True)
        
        # run
        cbegan_conv = CBEGAN_Classifier(generator, discriminator, classifier, data)
        cbegan_conv.train(sample_folder)
else:
    print('Wrong model')
