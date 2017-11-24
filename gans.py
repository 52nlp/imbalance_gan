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
from tfrecord import *

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
 
def sample_y(m, n): # 16 , y_dim , fig count
    y = np.zeros([m,n])
    for i in range(m):
        y[i, i%n] = 1
    #y[:,7] = 1
    #y[-1,0] = 1
    #print(y)
    return y

def concat(z,y):
    return tf.concat([z,y],1)
def five_2(confusion_matrix):
    TP = confusion_matrix[0,0] + confusion_matrix[0,1] + confusion_matrix[1,0] + confusion_matrix[1,1]
    FN = confusion_matrix[0,2] + confusion_matrix[0,3] + confusion_matrix[0,4] + confusion_matrix[1,2] + confusion_matrix[1,3] + confusion_matrix[1,4]
    FP = confusion_matrix[2,0] + confusion_matrix[2,1] + confusion_matrix[3,0] + confusion_matrix[3,1] + confusion_matrix[4,0] + confusion_matrix[4,1]
    TN = confusion_matrix[2,2] + confusion_matrix[2,3] + confusion_matrix[2,4] + confusion_matrix[3,2] + confusion_matrix[3,3] + confusion_matrix[3,4] + confusion_matrix[4,2] + confusion_matrix[4,3] + confusion_matrix[4,4]
    sensitivity = TP / (TP+FN)
    specificity = TN / (TN+FP)
    acc = (TP+TN) / (TP+FN+TN+FP)
    return acc, sensitivity, specificity

class GAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data_all, data_min, data_val):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data_all = data_all
        self.data_min = data_min
        self.data_val = data_val

        self.lr = tf.placeholder(tf.float32)

        self.lam = 1e-3
        self.gamma = 0.5
        self.k_curr = 0.0


        # data
        self.z_dim = self.data_all.z_dim
        self.y_dim = self.data_all.y_dim # condition
        self.size = self.data_all.size
        self.channel = self.data_all.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.k = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        # nets
        self.G_sample = self.generator(self.z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)
    
        self.C_real = self.classifier(self.X, is_training=self.is_training)
        self.C_fake = self.classifier(self.G_sample, is_training=self.is_training, reuse = True)

        self.lam = 10
        eps = tf.random_uniform([], minval=0., maxval=1.)#batch_size = 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = discriminator(self.X_inter, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam * tf.reduce_mean(grad_norm - 1.)**2    

        # loss
        C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))
        C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y))
        
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = - tf.reduce_mean(self.D_fake)*0.6 #+ C_fake_loss*0.4
        self.C_loss = C_real_loss*0.6 + C_fake_loss*0.4 
        #self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))  
        #self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]

        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.generator.vars)
        self.C_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.C_loss, var_list=self.classifier.vars)


        self.correct_prediction = tf.equal(tf.argmax(self.C_real, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        argmax_prediction = tf.argmax(self.C_real, 1)
        argmax_y = tf.argmax(self.y, 1)

        #self.TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        #self.TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        #self.FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        #self.FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)

        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(argmax_y, argmax_prediction, num_classes=self.y_dim)
        self.auc = tf.metrics.auc(argmax_y, argmax_prediction)
        #self.C_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.C_fake_loss, var_list=self.generator.vars)        

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train_classifier(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 32, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 2e-4

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    self.C_solver1,
                    feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                    )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_real_loss_curr = self.sess.run(
                        [self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b})
                print(epoch,C_real_loss_curr)
                #print('Iter: {}; C_real_loss: {:.4}'.format(epoch,  C_real_loss_curr))

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)
     

    def train(self, sample_folder, ckpt_dir, training_epoches = 100000, batch_size = 128, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        
        image, label = read_and_decode(self.data_all.filename, batch_size,self.size)
        image_min, label_min = read_and_decode(self.data_min.filename, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data_val.filename, batch_size,self.size)

        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            
            n_d = 80 if epoch < 30 or (epoch+1) % 500 == 0 else 5
            n_g = 1 if epoch < 30 or (epoch+1) % 500 == 0 else 1
            n_c = 1 if epoch < 1000 or (epoch+1) % 500 == 0 else 40

            # update D
            for _ in range(n_d):
                #X_b, y_b = self.data_min(batch_size)
                X_b, y_b = get_batch(self.sess, image_min, label_min, self.y_dim, batch_size)

                self.sess.run(
                    [self.D_solver], #clip
                    feed_dict={self.X: X_b, self.z: sample_z(X_b.shape[0], self.z_dim), self.lr: learning_rate})
            # update G
            for _ in range(n_g):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.z: sample_z(X_b.shape[0], self.z_dim), self.y: y_b, self.lr: learning_rate, self.is_training: True})
                
            # update C
            for _ in range(n_c):
                #X_b, y_b = self.data_all(batch_size)
                X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)
                self.sess.run(
                    self.C_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(X_b.shape[0], self.z_dim), self.lr: learning_rate, self.is_training: True})
            
            # save img, model. print loss
            if epoch % 500 == 0 or epoch <= 100:

                #train
                G_loss_curr, D_loss_curr, C_loss_curr, C_acc_curr ,confusion_matrix= self.sess.run(
                        [self.G_loss, self.D_loss, self.C_loss, self.accuracy, self.confusion_matrix],
                        feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(X_b.shape[0], self.z_dim), self.lr: learning_rate, self.is_training: False})

                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_loss: {:.4}; C_acc: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, C_loss_curr, C_acc_curr))
                print(confusion_matrix)

                #test
                X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
                C_acc_val, confusion_matrix_val = self.sess.run(    
                                [self.accuracy, self.confusion_matrix],
                                feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})
                print(C_acc_val)
                print(confusion_matrix_val)
                #acc, sens, spec= five_2(confusion_matrix_val)
                #print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

                if epoch % 500 == 0:
                    y_s = sample_y(16, self.y_dim)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data_min.data2fig(samples)
                    plt.savefig('{}/{}.png'.format(sample_folder, str(fig_count).zfill(3)), bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'GAN_C.ckpt', global_step=epoch)

    def test(self, sample_folder, sample_num):
        y_s = sample_y(sample_num, self.y_dim)
        samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(sample_num, self.z_dim)})
        for i, sample in enumerate(samples):
            new_sample = np.concatenate((sample,sample),axis = 2)
            new_sample = np.concatenate((new_sample,sample),axis = 2)
            #fig = self.data.data2fig(samples)
            plt.imshow(new_sample)
            plt.axis('off')
            plt.savefig('{}/{}_{}.png'.format(sample_folder, i%self.y_dim, str(i).zfill(3)), bbox_inches='tight')
            plt.close()


    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")


class WGAN():
    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        self.tanh = True

        self.z_dim = self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        # nets
        self.G_sample = self.generator(self.z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)

        # loss
        # improved wgan
        self.lam = 10
        eps = tf.random_uniform([], minval=0., maxval=1.)#batch_size = 64 or lower than 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = discriminator(self.X_inter, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam * tf.reduce_mean(grad_norm - 1.)**2
        

        
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = - tf.reduce_mean(self.D_fake)

        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)
        
        # clip
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 128, restore = True):
        i = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())
            #sess_glob.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        image, label = read_and_decode(self.data.filename, batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update D
            n_d = 100 if (restore==False and epoch < 25) or (epoch+1) % 500 == 0 else 5
            #print(n_d)

            for _ in range(n_d):
                #X_b, _ = self.data(batch_size)
                #self.sess.run(self.clip_D)
                #TFrecord
                X_b, _ = get_batch(self.sess, image, label, 1, batch_size)

                self.sess.run(
                        self.D_solver,
                        feed_dict={self.X: X_b, self.z: sample_z(X_b.shape[0], self.z_dim)}
                        )
            # update G
            self.sess.run(
                self.G_solver,
                feed_dict={self.z: sample_z(batch_size, self.z_dim)}
            )

            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_b, self.z: sample_z(X_b.shape[0], self.z_dim)})
                G_loss_curr = self.sess.run(
                        self.G_loss,
                        feed_dict={self.z: sample_z(X_b.shape[0], self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                if epoch % 500 == 0:
                    samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}.jpeg'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
            if epoch % 1000 == 0 and epoch != 0:
                self.saver.save(self.sess, ckpt_dir+'W_GAN.ckpt', global_step=epoch)
    def test(self, sample_folder, sample_num):
        samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(sample_num, self.z_dim)})
        for i, sample in enumerate(samples):
            #new_sample = np.concatenate((sample,sample),axis = 2) # for gray img
            #new_sample = np.concatenate((new_sample,sample),axis = 2)
            #fig = self.data.data2fig(samples)
            plt.imshow(sample)
            plt.axis('off')
            plt.savefig('{}/{}.jpeg'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
            plt.close()

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class Classifer():
    def  __init__(self, classifier, data, data_val):
        self.classifier = classifier
        self.data = data
        self.data_val = data_val

        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        self.logits = self.classifier(self.X, self.is_training)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        argmax_prediction = tf.argmax(self.logits, 1)
        argmax_y = tf.argmax(self.y, 1)

        # self.TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        # self.TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        # self.FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        # self.FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
        # self.precision = self.TP / (self.TP + self.FP)
        # self.recall = self.TP / (self.TP + self.FN)

        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(argmax_y, argmax_prediction, num_classes=self.y_dim)
        #self.auc = tf.metrics.auc(argmax_y, argmax_prediction)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cross_entropy,  var_list=self.classifier.vars)

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 1000000, batch_size = 256, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-4
        print(self.data.filename,self.data_val.filename)
        image, label = read_and_decode(self.data.filename, batch_size,self.size)
        image_val, label_val = read_and_decode(self.data_val.filename, batch_size,self.size)

        #image, label = read_and_decode('train_slm_512_01234.tfrecords',batch_size,self.size)
        #image_val, label_val = read_and_decode('test_slm_512_01234.tfrecords',batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)
        # for _ in range(1000):
        #     X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)
        #     X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
        #threads= tf.train.start_queue_runners(sess=self.sess)

        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                #X_b, y_b = self.data(batch_size)
                X_b, y_b = get_batch(self.sess, image, label, self.y_dim, batch_size)

                self.sess.run(
                    [self.train_step],
                    feed_dict={self.X: X_b, self.y: y_b, self.is_training: True})
            
            # save img, model. print loss
            if epoch % 200 == 0 or epoch <= 100:
                #C_loss_curr, C_acc_curr, TP, TN, FP, FN, precision, recall, confusion_matrix = self.sess.run(
                C_loss_curr, C_acc_curr, confusion_matrix = self.sess.run(	
                        #[self.cross_entropy, self.accuracy, self.TP, self.TN, self.FP, self.FN, self.precision, self.recall,self.confusion_matrix],
                        [self.cross_entropy, self.accuracy, self.confusion_matrix],
                        feed_dict={self.X: X_b, self.y: y_b, self.is_training: False})
                #print(epoch, C_loss_curr, C_acc_curr)
                #print('Iter: {}; C_loss: {:.4}; C_acc: {:.4}  TP: {}; TN: {:}; FP: {:}; FN: {:}; precision: {:}; recall: {:}'.format(epoch, C_loss_curr, C_acc_curr,TP,TN,FP,FN,precision,recall))
                print('Iter: {}; C_loss: {:.4}; C_acc: {:.4} '.format(epoch, C_loss_curr, C_acc_curr))

                #print(''.format(TP,TN,FP,FN,precision,recall))
                print(confusion_matrix)
                acc, sens, spec = five_2(confusion_matrix)
                print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

                # validation
                X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
                confusion_matrix_final = self.sess.run(
                            self.confusion_matrix,
                            feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})

                if epoch % 500 == 0 :
                    batch_number = self.data_val.len//batch_size
                    for _ in range(batch_number-1):
                        X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
                        confusion_matrix_val = self.sess.run(
                                    self.confusion_matrix,
                                    feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})

                        confusion_matrix_final+=confusion_matrix_val
                print(confusion_matrix_final)

                acc, sens, spec = five_2(confusion_matrix_final)
                print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)

    def test(self, data_val, batch_size = 128):
        image_val, label_val = read_and_decode(self.data_val.filename, batch_size,self.size)
        threads= tf.train.start_queue_runners(sess=self.sess)

        batch_number = self.data_val.len//batch_size
        X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
        confusion_matrix_final = self.sess.run(
                        self.confusion_matrix,
                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})
        for _ in range(batch_number-1):
            X_val_b, y_val_b = get_batch(self.sess, image_val, label_val, self.y_dim, batch_size)
            confusion_matrix_val = self.sess.run(
                        self.confusion_matrix,
                        feed_dict={self.X: X_val_b, self.y: y_val_b, self.is_training: False})

            confusion_matrix_final+=confusion_matrix_val
        print(confusion_matrix_final)
        acc, sens, spec = five_2(confusion_matrix_final)
        print('Accurancy: {:.4}; Sensitivity: {:.4}; Specificity: {:.4}'.format(acc, sens, spec))

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")            
