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
    observation = observation[35:195][::2,::2] / 255.0
    observation = np.array(observation).astype(np.float32)
    observation = np.rollaxis(observation, 2, 0)
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
        if optimizer_p is None: optimizer_p = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=0.1)
        if optimizer_v is None: optimizer_v = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=0.1)
        if optimizer_c is None: optimizer_c = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=0.1)

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

    def to_gpu(self):
        self._cnn_net.to_gpu()
        self._policy_net.to_gpu()
        self._value_net.to_gpu()
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

    def get_all_optimizers(self):
        return [self._optimizer_c, self._optimizer_p, self._optimizer_v]

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

    def set_optimizer_params(self, params):
        for _1, opt in enumerate(self.get_all_optimizers()):
            for item in opt._states.keys():
                opt._states[item]['ms'] = np.frombuffer(params[_1][item], dtype=ctypes.c_float).reshape(opt._states[item]['ms'].shape)
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


def run_process(process_id, shared_weight_list, shared_rmsprop_params, running_reward):
    env = gym.make("Pong-v0")
    gpu_on = 0   # FIXME: still problems for gpu
    observation = env.reset()
    print "starting_running_reward = %f" % running_reward.value
    shared_model = A3C()

    shared_model.set_all_weight_list([[np.frombuffer(item, dtype=ctypes.c_float) for item in weights] for weights in shared_weight_list])
    if args.shared_opt:
        shared_model.set_optimizer_params(shared_rmsprop_params)
        print "shared_opt enabled"

    model = copy.deepcopy(shared_model)

    if gpu_on:
        model.to_gpu()
        shared_model.to_gpu()

    model.cleargrads() 
    shared_model.cleargrads()
    render = args.render
    index_epoch = 0
    discount_rate = 0.99
    
    reward_sum = 0
    reward = 0
    input_history, action_label_history, reward_history, policy_history, \
                            value_history, policy_action_history = [], [], [], [], [], []
    action_label_sum = np.zeros(3)
    action_label_len = 0
    num_of_games = 0
    num_of_frames_in_input = 4
    input_data = np.zeros((1, num_of_frames_in_input * 3, 80, 80)).astype(np.float32)
    image_index = 0
    time_step_index = 0
    t_max = args.t_max
    sum_diff_p = 0
    sum_diff_v = 0
    entropy = 0

    while True:
        if render:
            env.render()

        observation_processed = process_observation_2(observation)
        # print observation_processed.shape, input_data.shape
        input_data = np.roll(input_data, -3, axis=1)
        input_data[0][-3:][:] = observation_processed
        # print input_data[0]
        # TODO: assert rolling is correct
        # print np.sum(input_data[0][11]), np.sum(input_data[0][0])
        
        if gpu_on:
            input_data = cuda.to_gpu(input_data)

        # input_history.append(input_data)
        output_state = model.get_state(input_data)
        output_prop = model.get_policy(output_state)[0]
        output_value = model.get_value(output_state)[0][0]
        policy_history.append(output_prop)
        value_history.append(output_value)
        action = np.random.choice(np.array([0, 2, 3]), size = 1, p=output_prop.data)
        action_label = int(max([action - 1, 0]))
        policy_action_history.append(output_prop[action_label])
        action_label_history.append(action_label)
        entropy += -0.01 * F.sum(output_prop * F.log(output_prop))

        if reward != 0:
            num_of_games += 1

        observation, reward, done, info = env.step(action)
        time_step_index += 1
        reward_history.append(reward)
        reward_sum += reward
        if (done) or time_step_index >= t_max:
            action_label_len += len(action_label_history)

            if index_epoch % 100 == 0 and index_epoch != 0 and process_id == 0 and done:
                pickle.dump(shared_model, open('excited_%d.pkl' % index_epoch, 'wb'))

            average_policy = np.mean([_2.data for _2 in policy_history], axis=0)
            average_value = np.mean([_2.data for _2 in value_history])
            # policy_history = policy_history[np.arange(time_step_index), action_label_history]
            # print value_history.data, policy_history.data
            # print policy_history, value_history, policy_history.data.shape, value_history.data.shape
            if (done):
                input_data = np.zeros((1, 4 * 3, 80, 80)).astype(np.float32)  # reset inputs
                initial_v_value = 0 
            else:
                initial_v_value = value_history[-1].data 

            discounted_reward_history = np.array(discount_rewards(np.array(reward_history), initial_v_value)).astype(np.float32)

            diff_p = 0; diff_v = 0
            for _1 in range(len(policy_action_history)):
                # print discounted_reward_history[_1], value_history[_1].data, policy_action_history[_1].data
                diff_p += (discounted_reward_history[_1] - value_history[_1].data) * F.log(policy_action_history[_1])
                diff_v += (discounted_reward_history[_1] - value_history[_1]) ** 2

            # diff_p = (discounted_reward_history - value_history.data) * F.log(policy_history) # FIXME: positive or negative?
            
            diff_p += entropy
            if args.reverse_grad:
                diff_p = - diff_p
            # diff_v = (Variable(discounted_reward_history) - value_history) ** 2
            # print diff_p.data.shape, diff_v.data.shape
            sum_diff_p += np.sum(diff_p.data)
            sum_diff_v += np.sum(diff_v.data)
            diff = F.sum(diff_p + diff_v * 0.5)   # FIXME: good to simply do sum?
            diff.backward()         # FIXME: 1. is grad accumulated?  2. does grad backprop to the CNN?
            # print model._cnn_net.conv_1.W.grad[0][0][0:3]
            # print model._policy_net.fully_conn_2.W.grad[0][:10]
            # print model._value_net.fully_conn_2.W.grad[0][:10]

            if args.train:
                with lock_1:
                    shared_model.cleargrads()
                    shared_model.set_all_grad_list(model.get_all_grad_list())
                    # print "process_id = %d" % process_id
                    # print np.frombuffer(shared_weight_list[0][0], dtype=ctypes.c_float)[0:10]
                    # print shared_model._cnn_net.conv_1.W.data[0][0][0]
                    # time.sleep(4)
                    shared_model.update()
                # print np.frombuffer(shared_rmsprop_params[0]['/conv_1/b'], dtype=ctypes.c_float)
                model = copy.deepcopy(shared_model)
                # model.set_all_weight_list(shared_model.get_all_weight_list())
                model.cleargrads()
            
            num_of_games = 0
            input_history, action_label_history, reward_history, policy_history, \
                            value_history, policy_action_history = [], [], [], [], [], []

            entropy = 0
            time_step_index = 0
            
            if done:
                with lock_2:
                    running_reward.value = running_reward.value * 0.99 + reward_sum * 0.01 
                print "process_id = %d, epoch #%d, reward_sum = %f, running_reward = %f, average_policy = %s, average_value = %s, num of frames = %d" % \
                        (process_id, index_epoch, reward_sum, running_reward.value, str(average_policy), str(average_value), action_label_len)
                print "average diff p = %f, average diff v = %f" % (sum_diff_p / action_label_len, sum_diff_v / action_label_len)
                time.sleep(0.5)
                sum_diff_p = 0; sum_diff_v = 0
                reward_sum = 0
                observation = env.reset()
                index_epoch += 1
                action_label_sum = np.zeros(3)
                action_label_len = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--starting_running_reward", type=float, default=-21.0)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume_file", type=str, default=None)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--reverse_grad", type=int, default=0)
    parser.add_argument("--t_max", type=int, default=5)
    parser.add_argument("--process_num", type=int, default=5)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--shared_opt", type=int, default=1)
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

    shared_rmsprop_params = [[]] * 3
    for _1, rms_optimizer in enumerate(shared_model.get_all_optimizers()):
        shared_rmsprop_params[_1] = {}
        for item in rms_optimizer._states.keys():
            shared_rmsprop_params[_1][item] = mp.RawArray(ctypes.c_float, rms_optimizer._states[item]['ms'].flatten())

    processes = [[]] * args.process_num
    for item in range(args.process_num):
        processes[item] = mp.Process(target=run_process, args=(item, shared_weight_list, shared_rmsprop_params, running_reward))

    for item in processes:
        item.start()
        
    for item in processes:
        item.join()

