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

parser = argparse.ArgumentParser()
# parser.add_argument("starting_running_reward", type=float)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--resume_file", type=str, default=None)
parser.add_argument("--render", type=int, default=0)
parser.add_argument("--reverse_grad", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=10)
args = parser.parse_args()

class Conv_NN(Chain):
    def __init__(self):
        super(Conv_NN, self).__init__(
            conv_1=L.Convolution2D(4, 4, 3, stride=2, pad=1),
            conv_2=L.Convolution2D(4, 4, 3, stride=2, pad=1),
            fully_conn_1 = L.Linear(1600,200),
            fully_conn_2 = L.Linear(200,3),
        )

    def __call__(self, x_data):
        output = Variable(x_data)
        output = F.relu(self.conv_1(output))
        output = F.relu(self.conv_2(output))
        output = F.sigmoid(self.fully_conn_1(output))
        output = F.softmax(self.fully_conn_2(output))
        return output

def process_observation(observation):
    # return observation[::2,::2,0] / 256.0
    observation = observation[35:195]
    observation[observation == 144] = 0  # erase background (background type 1)
    observation[observation == 109] = 0  # erase background (background type 2)
    observation[observation != 0] = 1  # everything else (paddles, ball) just set to 1
    observation = np.array([[observation[::2,::2,0]]]).astype(np.float32)
    return observation

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * 0.99 + r[t]
        discounted_r[t] = running_add
    
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r

def main():
    env = gym.make("Pong-v0")
    gpu_on = 0
    observation = env.reset()
    if args.resume_file is None:
        model = Conv_NN()
    else:
        model = pickle.load(open(args.resume_file, 'rb'))

    if gpu_on:
        model.to_gpu()

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

    while True:
        if render:
            env.render()
        observation_processed = process_observation(observation)
        input_data = np.roll(input_data, -1, axis=1)
        input_data[0][-1] = observation_processed[0]
        # print np.sum(input_data[0][3]), np.sum(input_data[0][1])
        # time.sleep(0.5)
        
        if gpu_on:
            input_data = cuda.to_gpu(input_data)

        output_prop = model(input_data)
        # action = 2 if np.random.uniform() < output_prop.data else 3  
        action = np.random.choice(np.array([0, 2, 3]), size = 1, p=output_prop.data[0])
        fake_label = np.zeros(3)
        fake_label[max([action - 1, 0])] = 1
        input_history.append(input_data)
        fake_label_history.append(fake_label)

        if reward != 0:
            # discount = float(1 - discount_rate) / (1 - discount_rate ** len(fake_label_history))
            # discount = 1
            num_of_games += 1
            fake_label_sum += sum(fake_label_history)
            fake_label_len += len(fake_label_history)
            previous_observation_processed = None
        
        observation, reward, done, info = env.step(action)
        reward_history.append(reward)
        reward_sum += reward
        if done:
            # print fake_label_sum, fake_label_len
            print "epoch #%d, reward_sum = %f, average_prop = %s" % \
                    (index_epoch, reward_sum, str(fake_label_sum / fake_label_len))

            if index_epoch % 100 == 0 and index_epoch != 0:
                pickle.dump(model, open('excited_%d.pkl' % index_epoch, 'wb'))
            
            if index_epoch % args.batch_size == 0 and index_epoch != 0:  
                print "average num of frames = %f" % (len(fake_label_history) / float(num_of_games))
                print "updating..."
                input_history = np.vstack(input_history).astype(np.float32)
                discounted_reward_history = np.array(discount_rewards(np.array(reward_history))).astype(np.float32)
                fake_label_history = np.vstack(fake_label_history).astype(np.float32)
                num_splits = 10
                len_of_each_split = input_history.shape[0] / num_splits
                for _1 in range(num_splits):
                    start_index = _1 * len_of_each_split
                    end_index = (_1 + 1) * len_of_each_split
                    output_prop = model(input_history[start_index:end_index])
                    diff =  np.multiply((output_prop.data - fake_label_history[start_index:end_index]), \
                            discounted_reward_history[start_index:end_index].reshape(len(fake_label_history[start_index:end_index]), 1))
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
    main()
