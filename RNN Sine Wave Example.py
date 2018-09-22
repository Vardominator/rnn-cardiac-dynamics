
# coding: utf-8

# # Sine Wave RNN Example
# 
# ### Authored by Varderes Barsegyan
# 
# ### 8/18/2018

# ### Import necessary libraries

# In[20]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Class for generating sine wave data and batches within the data

# In[2]:


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)
        
    def ret_true(self, x_series):
        return np.sin(x_series)
    
    def next_batch(self, batch_size, steps, future_steps=1, return_batch_ts=False):
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        
        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))
        
        # Create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0, steps + future_steps) * self.resolution
        
        # Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)
        
        # FORMATTING for RNN
        if return_batch_ts:
            return y_batch[:,:-future_steps].reshape(-1, steps, 1), y_batch[:,future_steps:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:,:-future_steps].reshape(-1, steps, 1), y_batch[:,future_steps:].reshape(-1, steps, 1)
        
        


# ### Generate data and plot

# In[3]:


ts_data = TimeSeriesData(250, 0, 10)
plt.plot(ts_data.x_data, ts_data.y_true)


# ### Generate batch

# In[4]:


num_time_steps = 60
y1,y2,ts = ts_data.next_batch(1, num_time_steps, 1, True)
plt.plot(ts.flatten()[1:], y2.flatten(),)


# ### Show batch on original wave

# In[5]:


plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single Training Instance')
plt.legend()
plt.tight_layout()


# ### Create a training instance for 20 steps into the future

# In[6]:


future_steps = 20

train_inst = np.linspace(5, 5 + ts_data.resolution*(num_time_steps+future_steps), num_time_steps+future_steps)
plt.title('A TRAINING INSTANCE')
plt.plot(train_inst[:-future_steps], ts_data.ret_true(train_inst[:-future_steps]), 'bo', markersize=15, alpha=0.5, label='INSTANCE')
plt.plot(train_inst[future_steps:], ts_data.ret_true(train_inst[future_steps:]), 'ko', markersize=7, label='TARGET')
plt.legend()


# ### Create the model and set up the TensorFlow graph

# In[7]:


tf.reset_default_graph()
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
num_train_iterations = 5000
batch_size = 1


# ### Tensorflow placeholders

# In[8]:


X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])


# ### RNN Cell Layer

# In[9]:


# GATED RECURRENT UNIT CELL
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)


# In[10]:


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


# ### Define the loss and optimizer

# In[11]:


# MEAN SQUARED ERROR LOSS
loss = tf.reduce_mean(tf.square(outputs - y))

# ADAM OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)


# ### Run TensorFlow

# In[12]:


init = tf.global_variables_initializer()
# USE IF GPUs AVAILABLE
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver()


# In[13]:


with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        
        sess.run(train, feed_dict={X:X_batch, y:y_batch})
        
        if iteration % 250 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print(iteration, "\tMSE", mse)
    
    saver.save(sess, "./rnn_time_series_model_vardo")


# ### Predict 20 step into the future

# In[14]:


with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_vardo")
    
    X_new = np.sin(np.array(train_inst[:-future_steps].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X:X_new})


# In[15]:


plt.title("TESTING THE MODEL")

# TRAINING INSTANCE
plt.plot(train_inst[:-future_steps], np.sin(train_inst[:-future_steps]), "bo", markersize=15, alpha=0.5, label='TRAINING INST')

# TARGET TO PREDICT(CORRECT TEST VALUES NP.SIN(TRAIN))
plt.plot(train_inst[future_steps:], np.sin(train_inst[future_steps:]), "ko", markersize=10, label='TARGET')

# MODELS PREDICTION
plt.plot(train_inst[future_steps:], y_pred[0, :, 0], 'r.', markersize=10, label='PREDICTIONS')

plt.xlabel('TIME')
plt.legend()
plt.tight_layout()


# ### Generating an entirely new sequence (zero sequence)

# In[16]:


with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_vardo")
    
    # SEED ZEROS
    zero_seq_seed = [0.0 for i in range(num_time_steps)]
    
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        
        zero_seq_seed.append(y_pred[0, -1, 0])
        


# In[17]:


plt.plot(ts_data.x_data, zero_seq_seed, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], 'r', linewidth=3)
plt.xlabel('TIME')
plt.ylabel('Y')


# ### This shows that the RNN, although fed with a seris of zeros, produces something similar to a sine wave. It has learned the general behavior.

# ### Let's now seed it with something that is the true sine value

# In[18]:


with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_vardo")
    
    # SEED
    training_instance = list(ts_data.y_true[:num_time_steps])
    
    for iteration in range(len(training_instance) - num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        
        training_instance.append(y_pred[0, -1, 0])


# In[19]:


plt.plot(ts_data.x_data, ts_data.y_true, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], training_instance, 'r', linewidth=3)
plt.xlabel('TIME')
plt.ylabel('Y')


# ### Almost exactly a sine wave!
