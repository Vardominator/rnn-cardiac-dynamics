import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class LorenzDataGenerator():
    def __init__(self, step_count=10000, dt=0.01, init_conds=(0., 1., 1.05)):
        self.step_count = step_count
        self.dt = dt
        self.times = np.arange(0.0, (self.step_count + 1)*self.dt, self.dt)
        self.xs = np.empty((self.step_count + 1, ))
        self.ys = np.empty((self.step_count + 1, ))
        self.zs = np.empty((self.step_count + 1, ))
        self.xs[0], self.ys[0], self.zs[0] = init_conds
        
        for i in range(self.step_count):
            x_dot, y_dot, z_dot = self.lorenz(self.xs[i], self.ys[i], self.zs[i])
            self.xs[i + 1] = self.xs[i] + (x_dot * self.dt)
            self.ys[i + 1] = self.ys[i] + (y_dot * self.dt)  
            self.zs[i + 1] = self.zs[i] + (z_dot * self.dt)              

    @staticmethod
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    def next_batch(self, batch_size, future_steps=10, return_batch_ts=False):
        rand_start_index = np.random.randint(0, len(self.times) - batch_size - future_steps)
        end_index = rand_start_index + batch_size + future_steps
        times = self.times[rand_start_index:end_index]

        # ONLY INTERESTED IN X-AXIS FOR NOW
        batch = self.xs[rand_start_index:end_index]
        # return batch[:-future_steps], batch[future_steps:], times
        if return_batch_ts:
            return batch[:-future_steps].reshape(-1, batch_size, 1), batch[future_steps:].reshape(-1, batch_size, 1), times
        else:
            return batch[:-future_steps].reshape(-1, batch_size, 1), batch[future_steps:].reshape(-1, batch_size, 1)


# GENERATE LORENZ DATA
lorenz = LorenzDataGenerator(1000, 0.01)

num_time_steps = 100
future_steps = 10

# GENERATE BATCH
x1, x2, t_batch = lorenz.next_batch(num_time_steps, future_steps, return_batch_ts=True)
plt.plot(t_batch.flatten()[future_steps:], x2.flatten(), )

# SHOW BATCH ON ORIGINAL WAVE
plt.plot(lorenz.times, lorenz.xs, 'b-', label='x(t)')
plt.plot(t_batch.flatten()[:-future_steps], x1.flatten(), 'k-')
plt.plot(t_batch.flatten()[future_steps:], x2.flatten(), 'r*', label='Single Training Instance')
plt.legend()
plt.tight_layout()



plt.show()