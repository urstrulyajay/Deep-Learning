import tensorflow as tf

class synthetic_data:
    
    def __init__(self,w,b,num_examples):
        
        self.w = w
        self.b = b
        self.num_examples = num_examples
        
    def generate_data(self):
        
        X = tf.zeros(shape=(self.num_examples,self.w.shape[0]))
        X += tf.random.normal(shape=X.shape)
        Y = tf.matmul(X,tf.reshape(self.w,(-1,1)))+self.b
        Y += tf.random.normal(shape=Y.shape,stddev=0.01)
        Y = tf.reshape(Y,(-1,1))
        return X,Y