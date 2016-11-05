import numpy as np
import cPickle as pickle
import gym, argparse, os, chainer
from datetime import datetime
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import copy
import time, ctypes
import multiprocessing as mp

def process_observation(observation):
    # return observation[::2,::2,0] / 256.0
    observation = observation[35:195]
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    observation = np.array([[observation[::2,::2,0]]]).astype(np.float32)
    return observation

def process_observation_2(observation):
    observation = observation[35:195][::2,::2,0] / 255.0
    observation = np.array(observation).astype(np.float32)
    return observation


def discount_rewards(r, initial_v_value):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = initial_v_value
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * 0.99 + r[t]
        discounted_r[t] = running_add
   
    return discounted_r

def get_all_weights(chain):
        return [item.data.flatten() for item in chain.params()]

def set_all_weights(chain, weight_list):
    for index, item in enumerate(chain.params()):
        assert (item.data.flatten().shape[0] == weight_list[index].shape[0])
        item.data = weight_list[index].reshape(item.data.shape)
    return

def get_all_grads(chain):
    return [item.grad.flatten() for item in chain.params()]

def set_all_grads(chain, grad_list):
    for index, item in enumerate(chain.params()):
        assert (item.data.flatten().shape[0] == grad_list[index].shape[0])
        item.grad = grad_list[index].reshape(item.data.shape)
    return

class A3C(object):
    def __init__(self, cnn_net=None, policy_net=None, value_net=None, 
                 optimizer_p=None, optimizer_v=None, optimizer_c= None):
        if cnn_net is None: cnn_net = CNN()
        in_channel = 200
        if policy_net is None: policy_net = Policy_net(in_channel)
        if value_net is None: value_net = Value_net(in_channel)
        if optimizer_p is None: optimizer_p = optimizers.RMSprop(lr=args.lr / args.batch_size, alpha=0.99, eps=0.1)
        if optimizer_v is None: optimizer_v = optimizers.RMSprop(lr=args.lr / args.batch_size, alpha=0.99, eps=0.1)
        if optimizer_c is None: optimizer_c = optimizers.RMSprop(lr=args.lr / args.batch_size, alpha=0.99, eps=0.1)

        self._cnn_net = cnn_net
        self._policy_net = policy_net
        self._value_net = value_net
        self._optimizer_p = optimizer_p
        self._optimizer_v = optimizer_v     # FIXME: do we need two optimizers?
        self._optimizer_c = optimizer_c
        self._optimizer_p.setup(self._policy_net)
        self._optimizer_v.setup(self._value_net)
        self._optimizer_c.setup(self._cnn_net)
        return

    def cleargrads(self):
        self._cnn_net.cleargrads()
        self._policy_net.cleargrads()
        self._value_net.cleargrads()
        return

    def get_state(self, input_data):
        return self._cnn_net(input_data)

    def get_policy(self, state_data):
        return self._policy_net(state_data)

    def get_value(self, state_data):
        return self._value_net(state_data)

    def update(self):
        # print "0"
        # print self._cnn_net.conv_1.W.data[0][0][0][0], self._policy_net.fully_conn_1.W.data[0][0], self._value_net.fully_conn_1.W.data[0][0]
        self._optimizer_c.update()
        # print "1"
        # print self._cnn_net.conv_1.W.data[0][0][0][0], self._policy_net.fully_conn_1.W.data[0][0], self._value_net.fully_conn_1.W.data[0][0]
        self._optimizer_p.update()
        # print "2"
        # print self._cnn_net.conv_1.W.data[0][0][0][0], self._policy_net.fully_conn_1.W.data[0][0], self._value_net.fully_conn_1.W.data[0][0]
        self._optimizer_v.update()
        # print "3"
        # print self._cnn_net.conv_1.W.data[0][0][0][0], self._policy_net.fully_conn_1.W.data[0][0], self._value_net.fully_conn_1.W.data[0][0]
        return

    def get_all_weight_list(self):
        return [get_all_weights(item) for item in [self._cnn_net, self._policy_net, self._value_net]]

    def set_all_weight_list(self, weight_list_list):
        set_all_weights(self._cnn_net, weight_list_list[0])
        set_all_weights(self._policy_net, weight_list_list[1])
        set_all_weights(self._value_net, weight_list_list[2])
        return

    def get_all_grad_list(self):
        return [get_all_grads(item) for item in [self._cnn_net, self._policy_net, self._value_net]]

    def set_all_grad_list(self, grad_list_list):
        set_all_grads(self._cnn_net, grad_list_list[0])
        set_all_grads(self._policy_net, grad_list_list[1])
        set_all_grads(self._value_net, grad_list_list[2])
        return


class CNN(Chain):
    def __init__(self, input_channel = 12):
        super(CNN, self).__init__(
            conv_1=L.Convolution2D(input_channel, 32, 8, stride=4),
            conv_2=L.Convolution2D(32, 32, 4, stride=2),
            # conv_3=L.Convolution2D(32, 64, 4, stride=1),
            fully_conn_1 = L.Linear(2048,200)
        )

    def __call__(self, x_data):
        output = Variable(x_data)
        output = F.relu(self.conv_1(output))
        output = F.relu(self.conv_2(output))
        # output = F.relu(self.conv_3(output))
        output = F.relu(self.fully_conn_1(output))
        # note that no pooling layers are included, since translation is important for most games
        return output

        
class Policy_net(Chain):
    def __init__(self, input_dim):
        super(Policy_net, self).__init__(
            fully_conn_2 = L.Linear(input_dim,3)
        )
    
    def __call__(self, input_data):
        output = F.softmax(self.fully_conn_2(input_data))
        return output
        

class Value_net(Chain):
    def __init__(self, input_dim):
        super(Value_net, self).__init__(
            fully_conn_2 = L.Linear(input_dim,1)
        )
    
    def __call__(self, input_data):
        output = self.fully_conn_2(input_data)
        return output


def run_process(process_id, shared_weight_list, running_reward):
    env = gym.make("Pong-v0")
    gpu_on = 0
    observation = env.reset()
    print "starting_running_reward = %f" % running_reward.value
    model = A3C()
    shared_model = A3C()

    shared_model.set_all_weight_list([[np.frombuffer(item, dtype=ctypes.c_float) for item in weights] for weights in shared_weight_list])

    model.set_all_weight_list(shared_model.get_all_weight_list())  # sync model with shared_model

    # if gpu_on:
    #     model.to_gpu()

    model.cleargrads() 
    shared_model.cleargrads()
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
    input_data = np.zeros((1, 4 * 3, 80, 80)).astype(np.float32)
    image_index = 0
    time_step_index = 0
    t_max = args.t_max
    sum_diff_p = 0
    sum_diff_v = 0

    while True:
        if render:
            env.render()
        observation_processed = process_observation_2(observation)
        input_data = np.roll(input_data, -3, axis=1)
        input_data[0][-3:][:] = observation_processed
        # print input_data.shape
        # print np.sum(input_data[0][11]), np.sum(input_data[0][0])
        # time.sleep(0.5)
        
        if gpu_on:
            input_data = cuda.to_gpu(input_data)

        input_history.append(input_data)
        output_prop = model.get_policy(model.get_state(input_data))
        action = np.random.choice(np.array([0, 2, 3]), size = 1, p=output_prop.data[0])
        action_label = int(max([action - 1, 0]))
        action_label_history.append(action_label)

        if reward != 0:
            num_of_games += 1
                
        observation, reward, done, info = env.step(action)
        time_step_index += 1
        reward_history.append(reward)
        reward_sum += reward
        if (done or reward == -1) or time_step_index >= t_max:
            action_label_len += len(action_label_history)

            if index_epoch % 100 == 0 and index_epoch != 0:
                pickle.dump(shared_model, open('excited_%d.pkl' % index_epoch, 'wb'))
    
            input_history = np.vstack(input_history).astype(np.float32)
            state_history = model.get_state(input_history)
            policy_history = model.get_policy(state_history)
            value_history = F.flatten(model.get_value(state_history))
            entropy = - 0.01 * F.sum(policy_history * F.log(policy_history), axis=1)  # discouraging premature convergence
            # print "policy: %s" % str(policy_history.data[:5])
            # print "value: %s" % str(value_history.data[:5])
            action_label_history = np.array(action_label_history)
            # print "action_label_history: %s" % str(action_label_history)
            average_policy = np.mean(policy_history.data, axis=0)
            average_value = np.mean(value_history.data)
            policy_history = policy_history[np.arange(time_step_index), action_label_history]
            # print value_history.data, policy_history.data
            # print policy_history, value_history, policy_history.data.shape, value_history.data.shape
            if (done or reward == -1):
                input_data = np.zeros((1, 4 * 3, 80, 80)).astype(np.float32)  # reset inputs
                initial_v_value = 0 
            else:
                initial_v_value = value_history[-1].data 

            discounted_reward_history = np.array(discount_rewards(np.array(reward_history), initial_v_value)).astype(np.float32)

            diff_p = (discounted_reward_history - value_history.data) * F.log(policy_history) # FIXME: positive or negative?
            
            diff_p += entropy
            if args.reverse_grad:
                diff_p = - diff_p
            diff_v = (Variable(discounted_reward_history) - value_history) ** 2
            # print diff_p.data.shape, diff_v.data.shape
            sum_diff_p += np.sum(diff_p.data)
            sum_diff_v += np.sum(diff_v.data)
            diff = F.sum(diff_p + diff_v * 0.5)   # FIXME: good to simply do sum?
            diff.backward()         # FIXME: 1. is grad accumulated?  2. does grad backprop to the CNN?
            # print model._cnn_net.conv_1.W.grad[0][0][0:3]
            # print model._policy_net.fully_conn_2.W.grad[0][:10]
            # print model._value_net.fully_conn_2.W.grad[0][:10]

            with lock_1:
                shared_model.cleargrads()
                shared_model.set_all_grad_list(model.get_all_grad_list())
                print "process_id = %d" % process_id
                print np.frombuffer(shared_weight_list[0][0])[0:10]
                # print shared_model._cnn_net.conv_1.W.data[0][0][0]
                # time.sleep(4)
                shared_model.update()

            model.cleargrads()
            model.set_all_weight_list(shared_model.get_all_weight_list())
            
            num_of_games = 0
            input_history, action_label_history, reward_history = [], [], []

            time_step_index = 0
            
            if done:
                with lock_2:
                    running_reward.value = running_reward.value * 0.99 + reward_sum * 0.01 
                print "process_id = %d" % process_id
                print "epoch #%d, reward_sum = %f, running_reward = %f, average_policy = %s, average_value = %s, num of frames = %d" % \
                        (index_epoch, reward_sum, running_reward.value, str(average_policy), str(average_value), action_label_len)
                print "average diff p = %f, average diff v = %f" % (sum_diff_p / action_label_len, sum_diff_v / action_label_len)
                sum_diff_p = 0; sum_diff_v = 0
                reward_sum = 0
                observation = env.reset()
                index_epoch += 1
                action_label_sum = np.zeros(3)
                action_label_len = 0

                # if index_epoch % args.batch_size == 0 and index_epoch != 0:
                #     print "updating..."
                #     model.update()
                #     model.cleargrads()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--starting_running_reward", type=float, default=-21.0)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume_file", type=str, default=None)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--reverse_grad", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--t_max", type=int, default=5)
    args = parser.parse_args()

    lock_1 = mp.Lock()
    lock_2 = mp.Lock()
    running_reward = mp.Value(ctypes.c_float, args.starting_running_reward)
    if args.resume_file is None:
        shared_model = A3C()
    else:
        shared_model = pickle.load(open(args.resume_file, 'rb'))

    shared_weight_list = [[mp.RawArray(ctypes.c_float, item) for item in weights] 
                                        for weights in shared_model.get_all_weight_list()]

    num_of_processes = 4
    processes = [[]] * num_of_processes
    for item in range(num_of_processes):
        processes[item] = mp.Process(target=run_process, args=(item, shared_weight_list, running_reward))

    for item in processes:
        item.start()
        
    for item in processes:
        item.join()

