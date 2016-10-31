import numpy as np
import cPickle as pickle
import gym, argparse, os, chainer
from datetime import datetime
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import copy
import time


def discount_rewards(r, initial_v_value):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = initial_v_value
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * 0.99 + r[t]
        discounted_r[t] = running_add
   
    return discounted_r

class A3C(object):
    def __init__(self, cnn_net=None, policy_net=None, value_net=None):
        if cnn_net is None: cnn_net = CNN()
        if policy_net is None: policy_net = Policy_net()
        if value_net is None: value_net = Value_net()

        self._cnn_net = cnn_net
        self._policy_net = policy_net
        self._value_net = value_net
        return

    def cleargrads():
    	self._cnn_net.cleargrads()
    	self._policy_net.cleargrads()
    	self._value_net.cleargrads()
    	return

   	def get_policy():
   		pass

   	def get_value():
   		pass


class CNN(Chain):
    def __init__(self):
        super(Conv_NN, self).__init__(
            conv_1=L.Convolution2D(4, 4, 8, stride=2, pad=1),
            conv_2=L.Convolution2D(4, 4, 8, stride=1, pad=1)
        )

    def __call__(self, x_data):
        output = Variable(x_data)
        output = F.relu(self.conv_1(output))
        output = F.relu(self.conv_2(output))
        # note that no pooling layers are included, since translation is important for most games
        return output


def process_observation(observation):
    # return observation[::2,::2,0] / 256.0
    observation = observation[35:195]
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    observation = np.array([[observation[::2,::2,0]]]).astype(np.float32)
    return observation

        
class Policy_net(Chain):
    def __init__(self):
        super(Conv_NN, self).__init__(
            fully_conn_1 = L.Linear(4356,200),
            fully_conn_2 = L.Linear(200,3)
        )
    
    def __call__(self, input_data):
        output = F.sigmoid(self.fully_conn_1(input_data))
        output = F.softmax(self.fully_conn_2(output))
        return output
        

class Value_net(Chain):
    def __init__(self):
        super(Conv_NN, self).__init__(
            fully_conn_1 = L.Linear(4356,200),
            fully_conn_2 = L.Linear(200,3)
        )
    
    def __call__(self, input_data):
        output = F.sigmoid(self.fully_conn_1(input_data))
        output = F.linear(self.fully_conn_2(output))
        return output


def main():
    env = gym.make("Pong-v0")
    gpu_on = 0
    observation = env.reset()
    running_reward = args.starting_running_reward
    print "starting_running_reward = %f" % running_reward
    if args.resume_file is None:
        model = A3C()
    else:
        model = pickle.load(open(args.resume_file, 'rb'))

    # if gpu_on:
    #     model.to_gpu()

    model.cleargrads() 
    render = args.render
    index_epoch = 0
    discount_rate = 0.99
    optimizer = optimizers.RMSprop(lr=args.lr / args.batch_size, alpha=0.99, eps=1e-08)
    optimizer.setup(model)
    reward_sum = 0
    reward = 0
    input_history, fake_label_history, reward_history = [], [], []
    previous_observation_processed = None
    fake_label_sum = np.zeros(3)
    fake_label_len = 0
    num_of_games = 0
    input_data = np.zeros((1, 4, 80, 80)).astype(np.float32)
    # input_data = np.array(range(256)).reshape(1,4,8,8)
    image_index = 0
    time_step_index = 0
    t_max = 5

    while True:
        if render:
            env.render()
        observation_processed = process_observation(observation)
        input_data = np.roll(input_data, -1, axis=1)
        input_data[0][-1][:] = observation_processed[0]
        # print np.sum(input_data[0][3]), np.sum(input_data[0][1])
        # time.sleep(0.5)
        
        if gpu_on:
            input_data = cuda.to_gpu(input_data)

        output_prop = model._policy_net(model._cnn_net(input_data))
        # action = 2 if np.random.uniform() < output_prop.data else 3  
        action = np.random.choice(np.array([0, 2, 3]), size = 1, p=output_prop.data[0])
        fake_label = np.zeros(3)
        fake_label[max([action - 1, 0])] = 1
        input_history.append(input_data)
        fake_label_history.append(fake_label)

        if reward != 0:
            num_of_games += 1
                
        observation, reward, done, info = env.step(action)
        time_step_index += 1
        reward_history.append(reward)
        reward_sum += reward
        if done or time_step_index > t_max:
            fake_label_sum += sum(fake_label_history)
            fake_label_len += len(fake_label_history)
            # print fake_label_sum, fake_label_len
            running_reward = running_reward * 0.99 + reward_sum * 0.01
            print "epoch #%d, reward_sum = %f, running_reward = %f, average_prop = %s" % \
                    (index_epoch, reward_sum, running_reward, str(fake_label_sum / fake_label_len))

            if index_epoch % 100 == 0 and index_epoch != 0:
                pickle.dump(model, open('excited_%d.pkl' % index_epoch, 'wb'))
            
            if index_epoch % args.batch_size == 0 and index_epoch != 0:  
                print "average num of frames = %f" % (len(fake_label_history) / float(num_of_games))
                print "updating..."
                input_history = np.vstack(input_history).astype(np.float32)
                initial_v_value = model._value_net(model._cnn_net(input_history[-1:]))
                discounted_reward_history = np.array(discount_rewards(np.array(reward_history), initial_v_value)).astype(np.float32)
                fake_label_history = np.vstack(fake_label_history).astype(np.float32)

                output_prop = model(input_history)
                diff =  np.multiply((output_prop.data - fake_label_history), \
                        discounted_reward_history.reshape(len(fake_label_history), 1))
                output_prop.grad = - diff if args.reverse_grad else diff
                output_prop.backward()         # grad is accumulated

                optimizer.update()
                model.cleargrads()
                num_of_games = 0
                input_history, fake_label_history, reward_history = [], [], []
                
            fake_label_sum = np.zeros(3)
            fake_label_len = 0
            reward_sum = 0
            observation = env.reset()
            index_epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--starting_running_reward", type=float, default=-21.0)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume_file", type=str, default=None)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--reverse_grad", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    main()
