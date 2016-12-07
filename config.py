"""configuration file"""

# num_of_frames_in_input = 2
# num_channels_in_each_frame = 3
# possible_actions = [0, 1, 2, 3, 4, 5]
# screen_range = [35, 195]
# in_channel = 256
# default_start_reward = 150
# game_name = "SpaceInvaders-v0"

# num_of_frames_in_input = 2
# num_channels_in_each_frame = 3
# possible_actions = [0, 2, 3]
# screen_range = [35, 195]
# in_channel = 256
# default_start_reward = -21.0
# game_name = "Pong-v0"
# preprocessing = False

num_of_frames_in_input = 2
num_channels_in_each_frame = 1
possible_actions = [0, 1, 2, 3]
screen_range = [35, 195]
in_channel = 256
default_start_reward = 1.7
game_name = "Breakout-v0"
preprocessing = False

####################################################

num_of_rows_in_screenshot = (screen_range[1] - screen_range[0]) / 2
step_interval_of_updating_lr = 1000000
learning_rate_annealing_factor = 0.7
