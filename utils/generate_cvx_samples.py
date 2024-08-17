import cvx_device
from utils import *

config = get_config_obj()

def get_device_samples(num_sample):
    num_device_list = config['params']['num_device']
    channels_device_3 = []
    channels_device_5 = []
    channels_device_7 = []
    channels_device_9 = []
    channels_device_11 = []
    channels_device_13 = []
    for num_device in num_device_list:
        cvx_device.return_channel_gain_hd


if __name__ == '__main__':
    get_device_samples(10000)