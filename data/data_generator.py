import numpy as np
from data.utils.LMMSE_estimator import *
from utils.utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
num_RIS = 10
num_device = 3

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

        train_dir = './raw/device/train'
        test_dir = './raw/device/test'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.save(train_dir + f'/train_combined_channel_device{num_device}.npy', train_data)
        np.save(test_dir + f'/test_combined_channel_device{num_device}.npy', test_data)


def generate_channel_power():
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

    print(f'power_bar general generate finish')

    # 划分数据集
    train_data, test_data = train_test_split(
        combined_channel.cpu(),
        test_size=0.2,
        random_state=1
    )

    train_dir = './raw/power/train'
    test_dir = './raw/power/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.save(train_dir + f'/train_combined_channel_power_general.npy', train_data)
    np.save(test_dir + f'/test_combined_channel_power_general.npy', test_data)


def generate_channel_RIS():
    num_RIS_list = config['params']['num_RIS']
    for num_RIS in num_RIS_list:
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

        print(f'num_RIS {num_RIS} generate finish')

        # 划分数据集
        train_data, test_data = train_test_split(
            combined_channel.cpu(),
            test_size=0.2,
            random_state=1
        )

        train_dir = './raw/RIS/train'
        test_dir = './raw/RIS/test'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.save(train_dir + f'/train_combined_channel_RIS{num_RIS}.npy', train_data)
        np.save(test_dir + f'/test_combined_channel_RIS{num_RIS}.npy', test_data)


def generate_channel_device_from_cvx():
    num_device_list = config['params']['num_device']  # get from config file
    for num_device in num_device_list:
        params_system = (num_antenna, num_RIS, num_device)

        # 生成信道数据
        combined_channel, _ = channel_generation_from_cvx(
            params_system,
            num_frame * num_device,
            noise_power_db,
            location_user,
            Rician_factor,
            channels_path='./from_cvx/device/channels',
            num_sample=10000,
            pilot_power=pilot_power,
            location_bs=location_bs,
            location_irs=location_irs,
            L_0=L_0,
            alpha=alpha
        )

        print(f'num_device {num_device} from cvx generate finish')

        # 划分数据集
        train_data, test_data = train_test_split(
            combined_channel.cpu(),
            test_size=0.2,
            random_state=1
        )

        train_dir = './from_cvx/device/train'
        test_dir = './from_cvx/device/test'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.save(train_dir + f'/train_combined_channel_device{num_device}.npy', train_data)
        np.save(test_dir + f'/test_combined_channel_device{num_device}.npy', test_data)


def generate_channel_power_from_cvx():
    params_system = (num_antenna, num_RIS, num_device)

    # 生成信道数据
    combined_channel, _ = channel_generation_from_cvx(
        params_system,
        num_frame * num_device,
        noise_power_db,
        location_user,
        Rician_factor,
        channels_path='./from_cvx/power/channels',
        num_sample=10000,
        pilot_power=pilot_power,
        location_bs=location_bs,
        location_irs=location_irs,
        L_0=L_0,
        alpha=alpha
    )

    print(f'power_bar general from cvx generate finish')

    # 划分数据集
    train_data, test_data = train_test_split(
        combined_channel.cpu(),
        test_size=0.2,
        random_state=1
    )

    train_dir = './from_cvx/power/train'
    test_dir = './from_cvx/power/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.save(train_dir + f'/train_combined_channel_power_general.npy', train_data)
    np.save(test_dir + f'/test_combined_channel_power_general.npy', test_data)


def generate_channel_ris_from_cvx():
    num_ris_list = config['params']['num_RIS']  # get from config file
    for num_RIS in num_ris_list:
        params_system = (num_antenna, num_RIS, num_device)

        # 生成信道数据
        combined_channel, _ = channel_generation_from_cvx(
            params_system,
            num_frame * num_device,
            noise_power_db,
            location_user,
            Rician_factor,
            channels_path='./from_cvx/ris/channels',
            num_sample=10000,
            pilot_power=pilot_power,
            location_bs=location_bs,
            location_irs=location_irs,
            L_0=L_0,
            alpha=alpha
        )

        print(f'num_ris {num_RIS} from cvx generate finish')

        # 划分数据集
        train_data, test_data = train_test_split(
            combined_channel.cpu(),
            test_size=0.2,
            random_state=1
        )

        train_dir = './from_cvx/ris/train'
        test_dir = './from_cvx/ris/test'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.save(train_dir + f'/train_combined_channel_ris{num_RIS}.npy', train_data)
        np.save(test_dir + f'/test_combined_channel_ris{num_RIS}.npy', test_data)


if __name__ == '__main__':
    # generate_channel_device()
    # generate_channel_power()
    # generate_channel_RIS()
    # generate_channel_device_from_cvx()
    # generate_channel_power_from_cvx()
    generate_channel_ris_from_cvx()
    pass