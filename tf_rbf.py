# !/usr/bin/env python
# encoding: utf-8

#from tf_kmeans import TFKMeans
from tensorflow.contrib.learn import KMeansClustering as tf_kmeans
import tensorflow as tf


class  TFRBFNet ( object ):
    '''
    TensorFlow 版的RBF+KMEANS
    其中RBF實現兩種訓練方式:
        - 使用梯度下降訓練隱藏曾到輸出的權重,類似BPNN
        - 直接使用Linear Regression, 求beta的線性方程解
    '''

    def  __init__ ( self , k , delta = 0.1 ):
        '''
            _delta: rbf的高斯擴展參數
            _beta: RBF層到輸出層的權重
            _input_n: 輸入神經元個數
            _hidden_num: 隱層神經元個數
            _output_n: 輸出神經元個數
            _max_iter: 迭代次數
            trainWithBeta: 第二種訓練方式
        '''
        #self._delta = tf.Variable(delta)
	self._delta = delta
        self._beta =  None
        self._input_n =  0
        self._hidden_num = k
        self._output_n =  0
        self.max_iter =  10
        self.trainWithBeta =  False
        self.sess = tf.Session()

    def  setup ( self , ni , nh , no ):
        '''
        網絡建立
        '''
        self._input_n = ni
        self._hidden_num = nh
        self._output_n = no

        self.input_layer = tf.placeholder(tf.float32, [ None , self ._input_n], name = 'inputs_layer' )
        self.output_layer = tf.placeholder(tf.float32, [ None , self ._output_n], name = 'outputs_layer' )
        self.getHiddenCenters( self._inputs)
        self.hidden_centers = tf.constant( self.hidden_centers, name = "hidden" )
	
	self.max_for_norm = tf.reduce_max(self.input_layer,axis=0)
	small_value = tf.fill(self.max_for_norm.shape,0.0000000001)
        self.max_for_norm = tf.where(tf.cast(self.max_for_norm,dtype=tf.bool), self.max_for_norm, small_value);

	self.hidden_layer =  self.rbfunction( self.input_layer, self.hidden_centers)


    def  fit ( self , inputs , outputs ):
        '''
        訓練
            inputs: 輸入數據
            ouputs: 輸出數據
        '''
        self._inputs = inputs
        self._outputs = outputs
        self.setup(inputs.shape[ 1 ], self ._hidden_num, outputs.shape[ 1 ])

        self.sess.run(tf.global_variables_initializer())
	print 'fit'
        if  self .trainWithBeta:
            self .LinearTrain()
        else :
            self .gradientTrain()


    def  LinearTrain ( self ):
        '''
        直接使用公式求解隱層到輸出層的beta參數，其中涉及到求逆操作
        '''
        beta = tf.matrix_inverse(tf.matmul(tf.transpose( self .hidden_layer), self .hidden_layer))
        beta_1 = tf.matmul(beta, tf.transpose( self .hidden_layer))
        beta_2 = tf.matmul(beta_1, self .output_layer)

        self ._beta =  self .sess.run(beta_2, feed_dict = { self .input_layer: self ._inputs, self .output_layer: self ._outputs})
        #預測輸出
        self .predictionWithBeta = tf.matmul( self .hidden_layer, self ._beta)


    def  gradientTrain ( self ):
        '''
        梯度下降法訓練RBF隱層->輸出層的參數
        '''
        self.trainWithBeta =  False
	print 'start_train'
        #最後預測的輸出
        self.predictionWithGD =  self.addLayer( self.hidden_layer, self._hidden_num, self ._output_n)
        #平方損失誤差
        self.loss = tf.reduce_mean(tf.square( self.predictionWithGD -  self.output_layer))
        #梯度下降優化
        self.optimizer = tf.train.GradientDescentOptimizer( 0.1 ).minimize( self .loss)
	print 'start_loop'
        self .sess.run(tf.global_variables_initializer())
        for i in  range ( self .max_iter):
            self .sess.run( self .optimizer, feed_dict = { self .input_layer: self ._inputs, self .output_layer: self ._outputs})
            if i % 1  ==  0 :
                print  ' iter: ' ,i, ' loss: ' , self .sess.run( self .loss, feed_dict = { self .input_layer: self ._inputs, self .output_layer: self ._outputs})


    def  predict ( self , inputs ):
        '''
        預測函數,根據不同的訓練方式，選擇不同的預測函數
        '''
        if  self .trainWithBeta:
            #直接計算beta
            return  self .sess.run( self .predictionWithBeta, feed_dict = { self .input_layer: inputs})
        else :
            #梯度下降方式訓練權重
            return  self .sess.run( self .predictionWithGD, feed_dict = { self .input_layer: inputs})

    def  addLayer ( self , inputs , inputs_size , output_size , activefunc = None ):
        '''
        添加隱層->輸出層
        只有一層，參數較少，沒有必要加正則以及Dropout
        '''

        self .weights = tf.Variable(tf.random_uniform([inputs_size, output_size], - 1.0 , 1.0 , tf.float32))
        self .biases = tf.Variable(tf.constant( 0.1 , tf.float32, shape = [ 1 ,output_size]))
        result = tf.matmul(inputs, self .weights)
        if activefunc is  None :
            return result
        else :
            return activefunc(result)

    def  getHiddenCenters ( self , inputs ):
        '''
        使用TF版的Kmeans基於歐式距離進行無監督聚類
        獲得中心
        '''
        kms = tf_kmeans(self._hidden_num,initial_clusters='kmeans_plus_plus')
	def input_for_kmeans():
	        data = tf.constant(inputs,tf.float32)
        	return (data,None)
	kms.fit(input_fn=input_for_kmeans,max_steps=100)
	self.hidden_centers = kms.clusters()
	print 'kmeans finished'
	#kms = TFKMeans( self._hidden_num, session = self.sess)
        #kms.train(tf.constant(inputs))
        #self.hidden_centers = kms.centers

    def  rbfunction ( self , x , c ):
	c = tf.divide(c, self.max_for_norm)
	x = tf.divide(x, self.max_for_norm )
        e_c = tf.expand_dims(c, 0 )
        e_x = tf.expand_dims(x, 1 )
        return tf.exp( - self._delta * tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(e_c, e_x)), 2 )))
