import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np
import random
from glob import glob

def pre_process(images): 
    #random_flip_up_down
    images = tf.image.random_flip_up_down(images) 
    #random_flip_left_right
    images = tf.image.random_flip_left_right(images) 
    #random_brightness
    images = tf.image.random_brightness(images, max_delta=0.3) 
    #random_contrast
    images = tf.image.random_contrast(images, 0.8, 1.2)
    #random_saturation
    tf.image.random_saturation(images, 0.3, 0.5)
    #new_size = tf.constant([image_size,image_size],dtype=tf.int32)
    #images = tf.image.resize_images(images, new_size)
    return images

def create_TFR(classes, class_num, filename, folder, img_size, is_val):
    data_list = []
    new_id = ''
    for i in range(class_num):
        class_sub = classes.split(',')[i]
        new_id += class_sub
        #print(class_sub)
        datapath = folder+class_sub+'/*' #HERE
        #print(datapath)
        data_list.extend(glob(datapath))

    #print(folder)
    #print(classes)
    label = [] 
    check = []
    label_count = -1
    for path in data_list: 
        #print(path)
        class_id = path.split('/')[6]#DR 6  #4 MNIST
        #print(class_id)
        if class_id not in check:
            check.append(class_id)
            label_count+=1
        label.append(label_count)

    #classes = {'1','2'} #人为 设定 2 类
    writer = tf.python_io.TFRecordWriter(filename) #要生成的文件

    tmp = list(zip(data_list, label))
    random.shuffle(tmp)
    random.shuffle(tmp)
    count = 0
    for img_path,index in tmp:
        #print(img_path)
        #down_sampling
        if is_val == False:
            r=0
            if index==0:
                r = random.randint(0,9)
            #if index==1:
             #   r = random.randint(0,49)
            if index==2:
                r = random.randint(0,5)
            
            if index==3 or index==4:#over_sampling
                for _ in range(3):
                    #img_path=class_path+img_name #每一个图片的地址
                    count+=1
                    img=Image.open(img_path)
                    img= img.resize((img_size,img_size))
                    img_raw=img.tobytes()#将图片转化为二进制格式
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    })) #example对象对label和image数据进行封装
                    writer.write(example.SerializeToString())  #序列化为字符串
            #if r==0:
            elif r==0:
                count+=1
                #img_path=class_path+img_name #每一个图片的地址
                img=Image.open(img_path)
                img= img.resize((img_size,img_size))
                #img = np.stack((img,)*3,axis=-1) # for grayscale to 3-channel 
                #print(img.shape)
                img_raw=img.tobytes()#将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })) #example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  #序列化为字符串
        else:
            count+=1
            #img_path=class_path+img_name #每一个图片的地址
            img=Image.open(img_path)
            img= img.resize((img_size,img_size))
            #img = np.stack((img,)*3,axis=-1) # for grayscale to 3-channel 
            #print(img.shape)
            img_raw=img.tobytes()#将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  #序列化为字符串


    writer.close()
    return count
def read_and_decode(filename,batch_size,img_size): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 3])  #reshape为128*128的3通道图片
    img = pre_process(img)
    img = tf.cast(img, tf.float32) * (1. / 255) #- 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    
    #one_hot = np.zeros((len(label),5))###self.y_dim  # my im gan = 2
    #for i,val in enumerate(label):
    #    one_hot[i,val]=1
    #label_onehot = one_hot

    img_b, label_b = tf.train.shuffle_batch([img, label],
                                   batch_size=batch_size, capacity=10000+3*batch_size,num_threads=64,
                                   min_after_dequeue=10000)

    return img_b, label_b


def get_batch(sess,image,label,classes_num,batch_size):
    #img_out, label_out = sess.run([image,label])
    X_b, y_b = sess.run([image,label])

    one_hot = np.zeros((batch_size,classes_num))###self.y_dim  # my im gan = 2
    for i,val in enumerate(y_b):
        one_hot[i,val]=1
    label_onehot = one_hot
    #print(X_b.shape,label_onehot.shape)

    return X_b, label_onehot


'''
#create_TFR('3,4',2)
image, label = read_and_decode('train_slm_512_01234.tfrecords',128)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

threads= tf.train.start_queue_runners(sess=sess)
for i in range(5):
    example, l = sess.run([image,label])#在会话中取出image和label
    
    img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
    img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        #print(example, l)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
image, label = read_and_decode('train_slm_512_01234.tfrecords',128)
image_val, label_val = read_and_decode('test_slm_512_01234.tfrecords',128)
threads= tf.train.start_queue_runners(sess=sess)
for _ in range(5000):
    img_out, label_out = sess.run([image,label])
    img_out_v, label_out_v = sess.run([image_val,label_val])
    print('train')
    print(label_out)
    print('test')
    print(label_out_v)
    
'''
