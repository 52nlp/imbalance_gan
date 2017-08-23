#test.py
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
    defect_id = sys.argv[3]
    defect_num = sys.argv[4]
    sample_num = sys.argv[5]
    
    print 'Model: '+model +'; Img_Resize: '+img_size +'; Defect_ID: '+defect_id +'; Defect_Num: '+defect_num +'; Sample_num: '+sample_num
    
    if model == 'wgan_adc' or model == 'lsgan_adc':
        if model == 'wgan_adc':
            loss_type = 'W'
        elif model == 'lsgan_adc':
            loss_type = 'LS'
        sample_folder = folder+'Samples_single/adc_'+img_size+'_'+defect_id+'_'+loss_type+'gan_conv'
        ckpt_folder = folder+'ckpt/'+loss_type+'_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/'+loss_type+'_GAN_'+img_size+'_'+defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        generator = G_conv(size=int(img_size),is_tanh=True)
        discriminator = D_conv(size=int(img_size))

        data = mydata(size=int(img_size), defect=defect_id, defect_num=1)
        
        # run
        GAN = WorLS_GAN(generator, discriminator, data)
        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))

    elif model == 'cwgan_adc' :#not finish
        
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
        generator = G_conv(size=int(img_size),is_tanh=False)
        discriminator = D_conv_condition(size=int(img_size))
        classifier = C_conv(size=int(img_size),class_num=int(defect_num))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num))
        
        # run
        GAN = CGAN_Classifier(generator, discriminator, classifier, data, loss_type = 'W')
        
        GAN.restore_ckpt(restore_folder)
    GAN.test(sample_folder,int(sample_num))
elif model == 'began_adc':
    
    sample_folder = folder+'Samples_single/adc_'+img_size+'_'+defect_id+'_'+'began_conv'
        ckpt_folder = folder+'ckpt/'+'BE_GAN_'+img_size+'_'+defect_id+'/'
        restore_folder = folder+'ckpt/'+'BE_GAN_'+img_size+'_'+defect_id+'/'
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
    generator = G_conv_BEGAN(size=int(img_size))
        discriminator = D_conv_BEGAN(size=int(img_size))
        
        data = mydata(size=int(img_size), defect=defect_id, defect_num=1)
        
        # run
        GAN = BEGAN(generator, discriminator, data, flag=False)
        GAN.restore_ckpt(restore_folder)
        GAN.test(sample_folder,int(sample_num))
elif model == 'c_adc':
    
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

data = mydata(size=int(img_size), defect=defect_id, defect_num=int(defect_num), test=True)
    
    # run
    c = Classifier(classifier, data)
        c.restore_ckpt(restore_folder)
        c.test()
    
    else: print('wrong model')

