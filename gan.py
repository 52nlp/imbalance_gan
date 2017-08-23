#gan.py
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

class CGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data, loss_type):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data = data
        self.loss_type = loss_type
        
        self.lr = tf.placeholder(tf.float32)
        
        
        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel
        
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
        eps = tf.random_uniform([tf.shape(self.G_sample)[0], 1, 1, 1], minval=0., maxval=1.)#batch_size = 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = self.discriminator(self.X_inter, self.y, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        grad_pen = tf.reduce_mean(self.lam * tf.square(grad_norm- 1.))
        # loss
        if self.loss_type == 'LS' :
            self.D_loss = 0.5 * (tf.reduce_mean((self.D_real - 1)**2) + tf.reduce_mean(self.D_fake**2))
            self.G_loss = 0.5 * tf.reduce_mean((self.D_fake - 1)**2)
        elif self.loss_type =='W' :
            self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake) + grad_pen
            self.G_loss = tf.reduce_mean(self.D_fake)
        
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
        
        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))
        
        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(self.G_loss + self.C_fake_loss, var_list=self.generator.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(self.C_real_loss , var_list=self.classifier.vars)
        #self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_fake_loss, var_list=self.generator.vars)
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def train_classifier(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 64, restore = True):
    fig_count = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        learning_rate = 2e-3
        
        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                              self.C_real_solver,
                              feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                              )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_real_loss_curr = self.sess.run(
                                                 [self.C_real_loss],
                                                 feed_dict={self.X: X_b, self.y: y_b})
                print('Iter: {}; C_real_loss: {:.4}'.format(epoch,  C_real_loss_curr))
            
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
            n_d = 100 if epoch < 10 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, y_b = self.data(batch_size)
                self.sess.run(self.clip_D)
                self.sess.run(
                              [self.D_solver],
                              feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim), self.lr: learning_rate}
                              )
            # update G
            for _ in range(1):
                self.sess.run(
                              self.G_solver,
                              feed_dict={self.y:y_b, self.z: sample_z(batch_size, self.z_dim), self.lr: learning_rate}
                              )
            # update C
            for _ in range(1):
                # real label to train C
                self.sess.run(
                              self.C_real_solver,
                              feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                              )

# fake img label to train G
'''for _ in range(3):
    self.sess.run(
    self.C_fake_solver,
    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})'''
        
        # save img, model. print loss
        if epoch % 100 == 0 or epoch < 100:
            D_loss_curr, C_real_loss_curr = self.sess.run(
                                                          [self.D_loss, self.C_real_loss],
                                                          feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr, C_fake_loss_curr = self.sess.run(
                                                              [self.G_loss, self.C_fake_loss],
                                                              feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                                                              print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:}; C_fake_loss: {:}'.format(epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr))
                                                              
                                                              if epoch % 500 == 0:
                                                                  y_s = sample_y(16, self.y_dim)
                                                                      samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})
                                                                          
                                                                          fig = self.data.data2fig(samples)
                                                                              plt.savefig('{}/{}.png'.format(sample_folder, str(fig_count).zfill(3)), bbox_inches='tight')
                                                                                  fig_count += 1
                                                                                      plt.close(fig)
                                                          
                                                          if epoch % 1000 == 0 and epoch != 0:
                                                                                      learning_rate = learning_rate/10
                                                                                      if self.loss_type == 'LS' :
                                                                                          self.saver.save(self.sess, ckpt_dir+'LS_GAN.ckpt', global_step=epoch)
                                                                                              elif self.loss_type == 'W' :
                                                                                                  self.saver.save(self.sess, ckpt_dir+'W_GAN.ckpt', global_step=epoch)
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

class CBEGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier, data):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.data = data
        self.lr = 2e-4
        
        self.lam = 1e-3
        self.gamma = 0.5
        self.k_curr = 0.0
        
        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel
        
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
        
        # loss BEGAN
        self.D_loss = tf.reduce_mean(self.D_real) - self.k * tf.reduce_mean(self.D_fake)
        self.G_loss = tf.reduce_mean(self.D_fake)
        
        
        
        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))
        
        # solver
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss + self.C_fake_loss, var_list=self.generator.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.C_real_loss, var_list=self.classifier.vars)
        #self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_fake_loss, var_list=self.generator.vars)
        
        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 32):
        fig_count = 0
        
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, y_b = self.data(batch_size)
                
                _, D_real_curr = self.sess.run(
                                               [self.D_solver, self.D_real],
                                               feed_dict={self.X: X_b, self.y:y_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr}
                                               )
            # update G
            for _ in range(1):
                _, D_fake_curr = self.sess.run(
                                               [self.G_solver, self.D_fake],
                                               feed_dict={self.y:y_b, self.z: sample_z(batch_size, self.z_dim)}
                                               )
            # update C
            for _ in range(1):
                # real label to train C
                self.sess.run(
                              self.C_real_solver,
                              feed_dict={self.X: X_b, self.y: y_b})
                    
                              # fake img label to train G
                              '''self.sess.run(
                                  self.C_fake_solver,
                                  feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})'''
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr, C_real_loss_curr = self.sess.run(
                                                              [self.D_loss, self.C_real_loss],
                                                              feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr})
                                                              G_loss_curr, C_fake_loss_curr = self.sess.run(
                                                                                                            [self.G_loss, self.C_fake_loss],
                                                                                                            feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                                                              print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr))
                                                              
                                                              if epoch % 100 == 0:
                                                                  y_s = sample_y(16, self.y_dim)
                                                                      samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})
                                                                          
                                                                          fig = self.data.data2fig(samples)
                                                                              plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
                                                                                  fig_count += 1
                                                                                      plt.close(fig)
                                                                                          self.k_curr = self.k_curr + self.lam * (self.gamma * D_real_curr - D_fake_curr)
                                                                                      #if epoch % 2000 == 0 and epoch != 0:
#self.saver.save(self.sess, os.path.join(ckpt_dir, "CBE_gan_classifier.ckpt"))
def restore_ckpt(self, ckpt_dir):
    self.saver.restore(self.sess, ckpt_dir)
        print("Model restored.")

class WorLS_GAN():
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
        eps = tf.random_uniform([tf.shape(self.G_sample)[0], 1, 1, 1], minval=0., maxval=1.)#batch_size = 64 or lower than 64
        self.X_inter = eps*self.X + (1. - eps)*self.G_sample
        self.D_tmp = self.discriminator(self.X_inter, reuse = True)
        grad = tf.gradients(self.D_tmp, self.X_inter)[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = self.lam *tf.reduce_mean(grad_norm- 1.)**2
        
        
        
        self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake) + grad_pen
        self.G_loss = tf.reduce_mean(self.D_fake)
        
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self.generator.vars)
        
        # clip
        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 64, restore = True):
        i = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())
        
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, _ = self.data(batch_size)
                self.sess.run(self.clip_D)
                self.sess.run(
                              self.D_solver,
                              feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)}
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
                                                          feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
                                  G_loss_curr = self.sess.run(
                                                              self.G_loss,
                                                              feed_dict={self.z: sample_z(batch_size, self.z_dim)})
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
                                                                      new_sample = np.concatenate((sample,sample),axis = 2)
                                                                      new_sample = np.concatenate((new_sample,sample),axis = 2)
                                                                      #fig = self.data.data2fig(samples)
                                                                      plt.imshow(new_sample)
                                                                      plt.axis('off')
                                                                      plt.savefig('{}/{}.jpg'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                                                                      plt.close()

def restore_ckpt(self, ckpt_dir):
    self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")

class BEGAN():
    def __init__(self, generator, discriminator, data, flag=True):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        
        self.lam = 1e-2
        self.gamma = 0.75
        self.k_curr = 0.0
        
        self.z_dim = self.data.z_dim
        self.size = self.data.size
        self.channel = self.data.channel
        
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.k = tf.placeholder(tf.float32)
        # nets
        if flag:
            self.G_sample = self.generator(self.z, reuse = True)
        
        else:
            self.G_sample = self.generator(self.z, reuse = False)
        self.D_real = self.discriminator(self.X, reuse = False)
        
        #self.D_real, _ = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)
        
        # loss
        self.D_loss = tf.reduce_mean(self.D_real) - self.k * tf.reduce_mean(self.D_fake)
        self.G_loss = tf.reduce_mean(self.D_fake)
        
        self.D_solver = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        
        
        if flag:
            tf.get_variable_scope().reuse_variables()
        self.G_solver = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)
        
        # clip
        
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#self.sess.run(tf.global_variables_initializer())

def train(self, sample_folder, ckpt_dir, training_epoches = 1000000, batch_size = 128, restore = True):
    i = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())


    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
        for epoch in range(training_epoches):
            # update D
            n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
            for _ in range(n_d):
                X_b, _ = self.data(batch_size)
                
                _, D_real_curr = self.sess.run(
                                               [self.D_solver, self.D_real],
                                               feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr}
                                               )
            # update G
            for _ in range(1):
                _, D_fake_curr = self.sess.run(
                                               [self.G_solver, self.D_fake],
                                               feed_dict={self.z: sample_z(batch_size, self.z_dim)}
                                               )
            
            # print loss. save images.
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                                            self.D_loss,
                                            feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim), self.k: self.k_curr})
                                            G_loss_curr = self.sess.run(
                                                                        self.G_loss,
                                                                        feed_dict={self.z: sample_z(batch_size, self.z_dim)})
                                            print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))
                                            
                                            if epoch % 500 == 0:
                                                samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})
                                                    
                                                    fig = self.data.data2fig(samples)
                                                        plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
                                                            i += 1
                                                                plt.close(fig)
            self.k_curr = self.k_curr + self.lam * (self.gamma * D_real_curr - D_fake_curr)
            if epoch % 1000 == 0 and epoch != 0:
                self.saver.save(self.sess, ckpt_dir+'BE_GAN.ckpt', global_step=epoch)
    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")
        return self.generator


class Classifier(object):
    def __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data
        
        self.lr = tf.placeholder(tf.float32)
        
        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim # condition
        self.size = self.data.size
        self.channel = self.data.channel
        
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        
        self.C_real, self.C_softmax = self.classifier(self.X)
        
        self.C_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y)) # real label
        
        # solver
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.9).minimize(self.C_real_loss , var_list=self.classifier.vars)
        
        self.saver = tf.train.Saver(max_to_keep=5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    def train_classifier(self, sample_folder, ckpt_dir, training_epoches = 5000, batch_size = 32, restore = True):
        fig_count = 0
        if not restore:
            self.sess.run(tf.global_variables_initializer())
        
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        learning_rate = 1e-3
        
        for epoch in range(training_epoches):
            # update C
            for _ in range(1):
                # real label to train C
                X_b, y_b = self.data(batch_size)
                self.sess.run(
                              self.C_real_solver,
                              feed_dict={self.X: X_b, self.y: y_b, self.lr: learning_rate}
                              )
            
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                C_real_loss_curr = self.sess.run(
                                                 [self.C_real_loss],
                                                 feed_dict={self.X: X_b, self.y: y_b})
                print('Iter: {}; C_real_loss: {}'.format(epoch,  C_real_loss_curr))
            
            if epoch % 500 == 0 and epoch != 0:
                learning_rate = learning_rate/10
                self.saver.save(self.sess, ckpt_dir+'classifier.ckpt', global_step=epoch)

def test(self):
    test_num = len(self.data.data)
        X_b, y_b = self.data(test_num)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        
        C_out = self.sess.run(self.C_softmax, feed_dict={self.X: X_b})
        
        C_real_loss = self.sess.run([self.C_real_loss], feed_dict={self.X: X_b, self.y: y_b})
        for y in range(test_num):
            if C_out[y][0]>C_out[y][1]:
                if y_b[y][0]==1:
                    TP += 1
                else:
                    FP += 1
            else:
                
                if y_b[y][0]==1:
                    FN += 1
                else:
                    TN += 1
        print(TP,FN)
        print(FP,TN)
        #print(y_b)
        #print(C_out)
        print(C_real_loss)

    def restore_ckpt(self, ckpt_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
        print("Model restored.")
