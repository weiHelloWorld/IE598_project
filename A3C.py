import numpy as np
import cPickle as pickle
import gym, argparse, os, chainer, datetime
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import copy
import time, ctypes
import multiprocessing as mp
from config import *

try:
    import cupy
except Exception as e:
    pass

num_of_rows_in_screenshot = (screen_range[1] - screen_range[0]) / 2
step_interval_of_updating_lr = 500000
learning_rate_annealing_factor = 0.7

def process_observation(observation):
    if num_channels_in_each_frame == 1:
        observation = observation[screen_range[0]:screen_range[1]][::2,::2,0] / 255.0
        observation = np.array(observation).astype(np.float32)
    elif num_channels_in_each_frame == 3:
        observation = observation[screen_range[0]:screen_range[1]][::2,::2] / 255.0
        observation = np.array(observation).astype(np.float32)
        observation = np.rollaxis(observation, 2, 0)
    
    return observation


def discount_rewards(r, initial_v_value):
    discounted_r = np.zeros_like(r)
    running_add = initial_v_value
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * 0.99 + r[t]
        discounted_r[t] = running_add
   
    return discounted_r

def get_all_weights(chain, in_cpu=True):
    temp = [item.data.flatten() for item in chain.params()]
    return np.concatenate(temp) if in_cpu else cupy.concatenate(temp)

def set_all_weights(chain, weight_list, start_index):
    # start_index = 0
    for item in chain.params():
        end_index = start_index + item.data.flatten().shape[0]
        item.data = weight_list[start_index: end_index].reshape(item.data.shape)
        start_index = end_index
    return end_index

def get_all_grads(chain):
    return [item.grad.flatten() for item in chain.params()]

def set_all_grads(chain, grad_list):
    for index, item in enumerate(chain.params()):
        assert (item.data.flatten().shape[0] == grad_list[index].shape[0])
        item.grad = cuda.to_cpu(grad_list[index]).reshape(item.data.shape)
    return

class A3C(object):
    def __init__(self, cnn_net=None, policy_net=None, value_net=None, 
                 optimizer_p=None, optimizer_v=None, optimizer_c= None):
        if cnn_net is None: cnn_net = CNN()
        if policy_net is None: policy_net = Policy_net(in_channel)
        if value_net is None: value_net = Value_net(in_channel)
        if optimizer_p is None: optimizer_p = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=0.1)
        if optimizer_v is None: optimizer_v = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=0.1)
        if optimizer_c is None: optimizer_c = optimizers.RMSprop(lr=args.lr, alpha=0.99, eps=0.1)

        self._cnn_net = cnn_net
        self._policy_net = policy_net
        self._value_net = value_net
        self._optimizer_p = optimizer_p
        self._optimizer_v = optimizer_v 
        self._optimizer_c = optimizer_c
        self._optimizer_p.setup(self._policy_net)
        self._optimizer_v.setup(self._value_net)
        self._optimizer_c.setup(self._cnn_net)
        return

    def set_learning_rate(self, learning_rate):
        for item in self.get_all_optimizers():
            item.lr = learning_rate
        return

    def get_learning_rate(self):
        return self._optimizer_c.lr

    def to_gpu(self, gpu_id=0):
        self._cnn_net.to_gpu(gpu_id)
        self._policy_net.to_gpu(gpu_id)
        self._value_net.to_gpu(gpu_id)
        return

    def zerograds(self):
        self._cnn_net.zerograds()
        self._policy_net.zerograds()
        self._value_net.zerograds()
        return

    def unchain_LSTM(self):
        self._cnn_net.lstm.h.unchain_backward()
        self._cnn_net.lstm.c.unchain_backward()
        return

    def reset_LSTM(self):
        self._cnn_net.lstm.reset_state()
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

    def in_cpu(self):
        return self._cnn_net._cpu

    def get_all_weight_list(self):
        temp = [get_all_weights(item, self.in_cpu()) for item in [self._cnn_net, self._policy_net, self._value_net]]
        return np.concatenate(temp) if self.in_cpu() else cupy.concatenate(temp)

    def set_all_weight_list(self, weight_list_list):
        start_index = 0
        for item in [self._cnn_net, self._policy_net, self._value_net]:
            start_index = set_all_weights(item, weight_list_list, start_index)
        assert (start_index == weight_list_list.shape[0])
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
    def __init__(self, input_channel = num_of_frames_in_input * num_channels_in_each_frame):
        super(CNN, self).__init__(
            conv_1=L.Convolution2D(input_channel, 32, 8, stride=4),
            conv_2=L.Convolution2D(32, 32, 4, stride=2),
            # conv_3=L.Convolution2D(32, 64, 4, stride=1),
            fully_conn_1 = L.Linear(512, in_channel),
            lstm=L.LSTM(in_channel, in_channel)
        )

    def __call__(self, x_data):
        output = Variable(x_data)
        output = F.relu(self.conv_1(output))
        output = F.max_pooling_2d(output, 2, 2)
        output = F.relu(self.conv_2(output))
        # output = F.relu(self.conv_3(output))
        output = F.relu(self.fully_conn_1(output))
        output = self.lstm(output)
        return output

        
class Policy_net(Chain):
    def __init__(self, input_dim):
        super(Policy_net, self).__init__(
            fully_conn_2 = L.Linear(input_dim,len(possible_actions))
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


def run_process(process_id, shared_weight_list, shared_rmsprop_params):
    env = gym.make(game_name)
    gpu_on = args.gpu_on   # FIXME: still problems for gpu
    observation = env.reset()
    shared_model = A3C()

    shared_model.set_all_weight_list(np.frombuffer(shared_weight_list, dtype=ctypes.c_float) )
    if args.shared_opt:
        shared_model.set_optimizer_params(shared_rmsprop_params)

    model = copy.deepcopy(shared_model)

    if gpu_on:
        model.to_gpu()
        # shared_model.to_gpu(1)

    model.zerograds() 
    shared_model.zerograds()
    render = args.render
    
    reward_sum = 0
    reward = 0
    input_history, action_label_history, reward_history, policy_history, \
                            value_history, policy_action_history = [], [], [], [], [], []
    action_label_len = 0
    # input_data = np.zeros((1, num_of_frames_in_input * 3, 80, 80)).astype(np.float32)
    input_data = np.zeros((1, num_of_frames_in_input * num_channels_in_each_frame, num_of_rows_in_screenshot, 80)).astype(np.float32)
    image_index = 0
    time_step_index = 0
    t_max = args.t_max
    sum_diff_p = 0
    sum_diff_v = 0
    entropy = 0
    last_checkpoint_time = time.time()
    previous_observation_processed = None
    accum_time = 0

    while True:
        # observation_processed = process_observation_2(observation)
        # # print observation_processed.shape, input_data.shape
        # input_data = np.roll(input_data, -3, axis=1)
        # input_data[0][-3:][:] = observation_processed
        # print input_data[0]
        # print np.sum(input_data[0][11]), np.sum(input_data[0][0])
        
        if gpu_on: input_data = cuda.to_gpu(input_data)

        # input_history.append(input_data)
        output_state = model.get_state(input_data)
        output_prop = model.get_policy(output_state)[0]
        output_value = model.get_value(output_state)[0][0]
        policy_history.append(output_prop)
        value_history.append(output_value)
        action_label = np.random.choice(np.array(range(len(possible_actions))), size = 1, p=cuda.to_cpu(output_prop.data))[0]
        action = possible_actions[action_label]
        policy_action_history.append(output_prop[action_label])
        action_label_history.append(action_label)
        entropy += -0.01 * F.sum(output_prop * F.log(output_prop))
        reward = 0
        for item in range(num_of_frames_in_input):
            observation, temp_reward, done, _ = env.step(action)
            reward_sum += temp_reward
            if temp_reward < -1:  # clip reward
                temp_reward = -1
            elif temp_reward > 1:
                temp_reward = 1

            if render: env.render()
            reward += temp_reward
            temp_observation_processed = process_observation(observation)
            if previous_observation_processed is None:
                observation_processed = temp_observation_processed
            else:
                observation_processed = np.maximum(temp_observation_processed, previous_observation_processed)
                
            previous_observation_processed = temp_observation_processed

            if gpu_on: observation_processed = cuda.to_gpu(observation_processed)
            input_data[0][item * num_channels_in_each_frame : (item + 1) * num_channels_in_each_frame][:] = observation_processed

        time_step_index += 1
        reward_history.append(reward)
        
        if (done) or time_step_index >= t_max:
            action_label_len += len(action_label_history)

            if time.time() - last_checkpoint_time > 1000 and done and process_id == 0:  # save every 1000 seconds
                filename = 'excited_%d.pkl' % accumulated_num_frames.value
                if os.path.isfile(filename):  # backup file if previous one exists
                    os.rename(filename, filename.split('.pkl')[0] + "_bak_" + datetime.datetime.now().strftime(
                        "%Y_%m_%d_%H_%M_%S") + '.pkl')
                pickle.dump(shared_model, open(filename, 'wb'))
                last_checkpoint_time = time.time()

            average_policy = np.mean([cuda.to_cpu(_2.data) for _2 in policy_history], axis=0)
            average_value = np.mean([cuda.to_cpu(_2.data) for _2 in value_history])
            # policy_history = policy_history[np.arange(time_step_index), action_label_history]
            # print value_history.data, policy_history.data
            # print policy_history, value_history, policy_history.data.shape, value_history.data.shape
            if (done):
                # input_data = np.zeros((1, 4 * 3, 80, 80)).astype(np.float32)  # FIXME: is it needed to reset inputs?
                initial_v_value = 0 
            else:
                initial_v_value = value_history[-1].data 

            discounted_reward_history = np.array(discount_rewards(np.array(reward_history), initial_v_value)).astype(np.float32)

            diff_p = 0; diff_v = 0
            for _1 in range(len(policy_action_history)):
                # print discounted_reward_history[_1], value_history[_1].data, policy_action_history[_1].data
                diff_p += (discounted_reward_history[_1] - value_history[_1].data) * F.log(policy_action_history[_1])
                diff_v += (discounted_reward_history[_1] - value_history[_1]) ** 2
            
            diff_p += entropy
            diff_p = - diff_p
            # diff_v = (Variable(discounted_reward_history) - value_history) ** 2
            # print diff_p.data.shape, diff_v.data.shape
            sum_diff_p += np.sum(diff_p.data)
            sum_diff_v += np.sum(diff_v.data)
            diff = F.sum(diff_p + diff_v * 0.5)   # FIXME: good to simply do sum?
            # end = time.time(); print "process_id = %d, 1: time = %f" % (process_id, end - start); start = time.time()
            # start = time.time()
            diff.backward()       
            model.unchain_LSTM()   # FIXME: where should we put this?  how often should we unchain?
            # end = time.time(); accum_time += (end-start)
            # end = time.time(); print "process_id = %d, 2: time = %f" % (process_id, end - start); start = time.time()
            # print model._cnn_net.conv_1.W.grad[0][0][0:3]
            # print model._policy_net.fully_conn_2.W.grad[0][:10]
            # print model._value_net.fully_conn_2.W.grad[0][:10]

            if args.train:
                with lock_1:
                    # shared_model.zerograds()
                    # with cuda.get_device(1):
                    # shared_model.set_all_weight_list(cuda.to_gpu(np.frombuffer(shared_weight_list, dtype=ctypes.c_float)))
                    shared_model.set_all_grad_list(model.get_all_grad_list())
                    # print "process_id = %d" % process_id
                    # print np.frombuffer(shared_weight_list[1][0], dtype=ctypes.c_float)[0:10]
                    # print shared_model._cnn_net.conv_1.W.data[0][0][0]
                    shared_model.update()
                    # shared_weight_list[:] = cuda.to_cpu(shared_model.get_all_weight_list())
                    
                # print np.frombuffer(shared_rmsprop_params[0]['/conv_1/b'], dtype=ctypes.c_float)
                # model = copy.deepcopy(shared_model)
                model.set_all_weight_list(cuda.to_gpu(shared_model.get_all_weight_list()))
                model.zerograds()
            
            input_history, action_label_history, reward_history, policy_history, \
                            value_history, policy_action_history = [], [], [], [], [], []

            entropy = 0
            time_step_index = 0
            
            if done:
                running_reward.value = running_reward.value * 0.99 + reward_sum * 0.01 
                accumulated_num_frames.value += action_label_len
                if running_reward.value > max_running_reward.value:
                    max_running_reward.value = running_reward.value
                    step_index_of_max_running_reward.value = accumulated_num_frames.value
                elif accumulated_num_frames.value - step_index_of_max_running_reward.value > step_interval_of_updating_lr:
                    step_index_of_max_running_reward.value = accumulated_num_frames.value
                    current_learning_rate.value *= learning_rate_annealing_factor

                if shared_model.get_learning_rate() > current_learning_rate.value:
                    shared_model.set_learning_rate(current_learning_rate.value)
                    print "new lr = %f" % shared_model.get_learning_rate()

                print "process_id = %d, step #%d, reward_sum = %f, running_reward = %f, average_policy = %s, average_value = %s, num of frames = %d" % \
                        (process_id, accumulated_num_frames.value, reward_sum, running_reward.value, str(average_policy), str(average_value), action_label_len)
                # print "average diff p = %f, average diff v = %f" % (sum_diff_p / action_label_len, sum_diff_v / action_label_len)
                # print "accum_time = %f" % accum_time; accum_time = 0
                # print datetime.datetime.now()
                print "time per M step = %f h" % ((time.time() - start_time ) * 1000000 / 3600 / (accumulated_num_frames.value - args.start_step))
                print max_running_reward.value, step_index_of_max_running_reward.value, current_learning_rate.value
                # time.sleep(0.1)
                sum_diff_p = 0; sum_diff_v = 0
                reward_sum = 0
                observation = env.reset()
                model.reset_LSTM()   # FIXME
                action_label_len = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_reward", type=float, default=default_start_reward)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume_file", type=str, default=None)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--t_max", type=int, default=5)
    parser.add_argument("--process_num", type=int, default=5)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--shared_opt", type=int, default=1)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--gpu_on", type=int,default=0)
    parser.add_argument("--record", type=int,default=0, help="record performance")
    parser.add_argument("--folder_to_record_result", type=str, default='/tmp/temp')
    args = parser.parse_args()

    start_time = time.time()

    lock_1 = mp.Lock()
    running_reward = mp.Value(ctypes.c_float, args.start_reward)
    max_running_reward = mp.Value(ctypes.c_float, args.start_reward)
    current_learning_rate = mp.Value(ctypes.c_float, args.lr)
    step_index_of_max_running_reward = mp.Value(ctypes.c_int, args.start_step)
    accumulated_num_frames = mp.Value(ctypes.c_int, args.start_step)

    print "start_reward = %f" % running_reward.value
    if args.resume_file is None:
        shared_model = A3C()
    else:
        shared_model = pickle.load(open(args.resume_file, 'rb'))
        print "model %s loaded" % (args.resume_file)

    shared_weight_list = mp.RawArray(ctypes.c_float, shared_model.get_all_weight_list())

    shared_rmsprop_params = [[]] * 3
    for _1, rms_optimizer in enumerate(shared_model.get_all_optimizers()):
        shared_rmsprop_params[_1] = {}
        for item in rms_optimizer._states.keys():
            shared_rmsprop_params[_1][item] = mp.RawArray(ctypes.c_float, rms_optimizer._states[item]['ms'].flatten())

    if not args.record:
        processes = [[]] * args.process_num
        for item in range(args.process_num):
            processes[item] = mp.Process(target=run_process, 
                args=(item, shared_weight_list, shared_rmsprop_params))

        for item in processes:
            item.start()
            
        for item in processes:
            item.join()
    else:
        env = gym.make(game_name)
        folder_to_record_result = args.folder_to_record_result
        if os.path.exists(folder_to_record_result):
            os.rename(folder_to_record_result, folder_to_record_result + str(datetime.datetime.now().strftime(
                        "%Y_%m_%d_%H_%M_%S")))

        env.monitor.start(folder_to_record_result)
        for _ in range(100):
            observation = env.reset()
            done = False
            input_data = np.zeros((1, num_of_frames_in_input * num_channels_in_each_frame, num_of_rows_in_screenshot, 80)).astype(np.float32)
            while True:
                output_state = shared_model.get_state(input_data)
                output_prop = shared_model.get_policy(output_state)[0]
                action_label = np.random.choice(np.array(range(len(possible_actions))), size = 1, p=(output_prop.data))[0]
                action = possible_actions[action_label]
                shared_model.unchain_LSTM()
                for item in range(num_of_frames_in_input):
                    if not done:
                        observation, temp_reward, done, _ = env.step(action)
                        observation_processed = process_observation(observation)
                        input_data[0][item * num_channels_in_each_frame : (item + 1) * num_channels_in_each_frame][:] = observation_processed

                if done: break

        env.monitor.close()
        gym.upload(folder_to_record_result, api_key='sk_JY6Go9lgRT2ebaOtfe6Lw')

