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

class GAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data_all, data_min):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data_all = data_all
        self.data_min = data_min


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

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))

        self.D_real = self.discriminator(self.X, self.y)
        self.D_fake = self.discriminator(self.G_sample, self.y, reuse = True)
    
        self.C_real = self.classifier(self.X)
        self.C_fake = self.classifier(self.G_sample, reuse = True)

        self.lam = 10
        eps = tf.random_uniform([], minval=0., maxval=1.)#batch_size = 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = discriminator(self.X_inter, self.y, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam * tf.reduce_mean(grad_norm - 1.)**2    

        # loss
        self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = - tf.reduce_mean(self.D_fake)

        #self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]

        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))  
        
        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.generator.vars)
        self.C_solver1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.C_real_loss + self.C_fake_loss, var_list=self.classifier.vars)
        self.C_solver2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.C_fake_loss, var_list=self.generator.vars)        

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
     

    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 64, restore = True):
        fig_count = 0  
        if not restore:
            self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 2e-4

        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5

            for _ in range(n_d):
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    [self.D_solver],#clip
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(y_b.shape[0], self.z_dim), self.lr: learning_rate}
                    )
            # update G
            for _ in range(1):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.y:y_b, self.z: sample_z(y_b.shape[0], self.z_dim), self.lr: learning_rate}
                )
                
            # update C
            # real label to train C
            for _ in range(10):
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                    self.C_solver1,
                    feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                    )
            
            # fake img label to train G
            for _ in range(10):
                self.sess.run(
                    self.C_solver2,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr, C_real_loss_curr = self.sess.run(
                        [self.D_loss, self.C_real_loss],
                        feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(y_b.shape[0], self.z_dim)})
                G_loss_curr, C_fake_loss_curr = self.sess.run(
                        [self.G_loss, self.C_fake_loss],
                        feed_dict={self.y: y_b, self.z: sample_z(y_b.shape[0], self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr))

                if epoch % 500 == 0:
                    y_s = sample_y(16, self.y_dim)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
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
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if (restore==False and epoch < 25) or (epoch+1) % 500 == 0 else 5
            #print(n_d)

            for _ in range(n_d):
                X_b, _ = self.data(batch_size)
                #self.sess.run(self.clip_D)
                #X_b, _ = get_batch(batch_size)#TFrecord
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
                    plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
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
            plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
            plt.close()

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class Classifer():
    def  __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data

        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.is_training = tf.placeholder(tf.bool, name='IsTraining')

        self.logits = self.classifier(self.X, self.is_training)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.cross_entropy)

        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, ckpt_dir, training_epoches = 1000000, batch_size = 128, restore = True):
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
                    [self.train_step],
                    feed_dict={self.X: X_b, self.y: y_b, self.is_training: True}
                    )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_loss_curr, C_acc_curr = self.sess.run(
                        [self.cross_entropy,self.accuracy],
                        feed_dict={self.X: X_b, self.y: y_b, self.is_training: True})
                #print(epoch, C_loss_curr, C_acc_curr)
                print('Iter: {}; C_loss: {:.4}; C_acc: {:.4}'.format(epoch, C_loss_curr, C_acc_curr))

            if epoch % 1000 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)
