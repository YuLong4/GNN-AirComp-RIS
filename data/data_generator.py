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

def check_data():
    train_data = np.load('./raw/device/train/train_combined_channel_device50.npy')
    test_data = np.load('./raw/device/test/test_combined_channel_device50.npy')
    
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    
    print(f'Train data statistics: {np.mean(train_data)}, {np.std(train_data)}')
    print(f'Test data statistics: {np.mean(test_data)}, {np.std(test_data)}')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_data.flatten(), bins=50)
    plt.title('Train Data Distribution')
    plt.subplot(1, 2, 2)
    plt.hist(test_data.flatten(), bins=50)
    plt.title('Test Data Distribution')
    plt.show()
    
    print(f'NaN in train data: {np.isnan(train_data).sum()}')
    print(f'NaN in test data: {np.isnan(test_data).sum()}')
    print(f'Inf in train data: {np.isinf(train_data).sum()}')
    print(f'Inf in test data: {np.isinf(test_data).sum()}')


if __name__ == '__main__':
    # generate_channel_device()
    # generate_channel_power()
    # generate_channel_RIS()
    check_data()