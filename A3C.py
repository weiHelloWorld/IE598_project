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
    def __init__(self, cnn_net=None, policy_net=None, value_net=None, 
                 optimizer_p=None, optimizer_v=None):
        if cnn_net is None: cnn_net = CNN()
        if policy_net is None: policy_net = Policy_net()
        if value_net is None: value_net = Value_net()
        if optimizer_p is None: optimizer_p = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=1e-08)
        if optimizer_v is None: optimizer_v = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=1e-08)

        self._cnn_net = cnn_net
        self._policy_net = policy_net
        self._value_net = value_net
        self._optimizer_p = optimizer_p
        self._optimizer_v = optimizer_v     # FIXME: do we need two optimizers?
        self._optimizer_p.setup(self._policy_net)
        self._optimizer_v.setup(self._value_net)
        return

    def cleargrads(self):
        self._cnn_net.cleargrads()
        self._policy_net.cleargrads()
        self._value_net.cleargrads()
        return

    def get_policy(self, input_data):
        return self._policy_net(self._cnn_net(Variable(input_data)))

    def get_value(self, input_data):
        return self._value_net(self._cnn_net(Variable(input_data)))

    def update(self):
        self._optimizer_p.update()
        self._optimizer_v.update()
        return


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
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
        super(Policy_net, self).__init__(
            fully_conn_1 = L.Linear(4356,200),
            fully_conn_2 = L.Linear(200,3)
        )
    
    def __call__(self, input_data):
        output = F.sigmoid(self.fully_conn_1(input_data))
        output = F.softmax(self.fully_conn_2(output))
        return output
        

class Value_net(Chain):
    def __init__(self):
        super(Value_net, self).__init__(
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
    
    reward_sum = 0
    reward = 0
    input_history, action_label_history, reward_history = [], [], []
    previous_observation_processed = None
    action_label_sum = np.zeros(3)
    action_label_len = 0
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

        print input_data
        output_prop = model.get_policy(input_data)
        action = np.random.choice(np.array([0, 2, 3]), size = 1, p=output_prop.data[0])
        action_label = np.zeros(3)
        action_label[max([action - 1, 0])] = 1
        input_history.append(input_data)
        action_label_history.append(action_label)

        if reward != 0:
            num_of_games += 1
                
        observation, reward, done, info = env.step(action)
        time_step_index += 1
        reward_history.append(reward)
        reward_sum += reward
        if done or time_step_index > t_max:
            action_label_sum += sum(action_label_history)
            action_label_len += len(action_label_history)
            # print action_label_sum, action_label_len
            running_reward = running_reward * 0.99 + reward_sum * 0.01
            print "epoch #%d, reward_sum = %f, running_reward = %f, average_prop = %s" % \
                    (index_epoch, reward_sum, running_reward, str(action_label_sum / action_label_len))

            if index_epoch % 100 == 0 and index_epoch != 0:
                pickle.dump(model, open('excited_%d.pkl' % index_epoch, 'wb'))
            
            # if index_epoch % args.batch_size == 0 and index_epoch != 0:  
            #     print "average num of frames = %f" % (len(action_label_history) / float(num_of_games))
            #     print "updating..."
            input_history = np.vstack(input_history).astype(np.float32)
            policy_history = model.get_policy(input_history)
            value_history = model.get_value(input_history)
            action_label_history = np.vstack(action_label_history).astype(np.float32).astype(bool)
            policy_history = policy_history[action_label_history]
            value_history = value_history[action_label_history]  
            print policy_history, value_history, policy_history.data.shape, value_history.data.shape
            initial_v_value = 0 if done else value_history[-1].data 
            discounted_reward_history = np.array(discount_rewards(np.array(reward_history), initial_v_value)).astype(np.float32)

            diff_p = (discounted_reward_history - value_history.data) * F.log(policy_history)
            diff_v = (Variable(discounted_reward_history) - value_history) ** 2
            diff = F.sum(diff_p + diff_v * 0.5)
            diff.backward()         # grad is accumulated

            model.update()
            model.cleargrads()
            num_of_games = 0
            input_history, action_label_history, reward_history = [], [], []
                
            action_label_sum = np.zeros(3)
            action_label_len = 0
            reward_sum = 0
            if done:
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
