import numpy as np
from data.utils.LMMSE_estimator import *
from utils.utils import *
from sklearn.model_selection import train_test_split

pilot_power = 10
noise_power_db = -70
Rician_factor = 10
L_0 = -25
num_frame = 4
alpha = [4, 1.8, 2.1]
location_bs = np.array([-200, 0, 30])
location_irs = np.array([0, 0, 10])
location_user = None

num_antenna = 1
num_RIS = 40
num_device = 10

config = get_config_obj()

combined_channel = None


def generate_channel_device():
    num_device_list = config['params']['num_device']  # get from config file
    for num_device in num_device_list:
        params_system = (num_antenna, num_RIS, num_device)

        # 生成信道数据
        combined_channel, _ = channel_generation(
            params_system,
            num_frame * num_device,
            noise_power_db,
            location_user,
            Rician_factor,
            num_sample=10000,
            pilot_power=pilot_power,
            location_bs=location_bs,
            location_irs=location_irs,
            L_0=L_0,
            alpha=alpha
        )

        print(f'num_device {num_device} generate finish')

        # 划分数据集
        train_data, test_data = train_test_split(
            combined_channel.cpu(),
            test_size=0.2,
            random_state=1
        )

        np.save(f'./raw/device/train/train_combined_channel_device{num_device}.npy', train_data)
        np.save(f'./raw/device/test/test_combined_channel_device{num_device}.npy', test_data)


if __name__ == '__main__':
    generate_channel_device()
