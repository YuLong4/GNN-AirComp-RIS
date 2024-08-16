import os.path

import torch.nn
from torch import nn, optim
from models.graph_net import Graph_net
from data.datasets import CustomDataset
from torch.utils.data import DataLoader
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

num_antenna = 1
num_RIS = 10
num_device = 10

mean_class = np.load('./save_model/save_mean_variance/mean_class_12dim.npy', allow_pickle=True)
var_class = np.load('./save_model/save_mean_variance/var_class_12dim.npy', allow_pickle=True)
num_class = 4
num_feature = 12
var_comm_noise = 1  # Variance of shadow fading, sigma_{0}^{2}
feature_noise_var = 0.4


def train(GNN, train_loader, optimizer, epochs, num_device, p_bar, nu, power_bar):
    batch_size = train_loader.batch_size
    loss_history = []
    for epoch in range(epochs):
        print(f'power_bar {power_bar} epoch {epoch + 1} begin:')
        total_loss = 0

        for batch_combined_channel in train_loader:
            batch_combined_channel = batch_combined_channel.cuda()
            input = (batch_combined_channel[:, :, 0].view(batch_size, 1, num_device) +
                     batch_combined_channel[:, :, 1:].sum(-1).view(batch_size, 1, num_device))
            channel_compose = torch.cat((input.real, input.imag), 1).cuda()
            Net_output = GNN.forward(channel_compose).clone()
            phase_shift_real = torch.cos(Net_output[:, num_device:num_device + num_RIS] * np.pi * 2)
            phase_shift_imag = torch.sin(Net_output[:, num_device:num_device + num_RIS] * np.pi * 2)
            batch_channel_coefficient_real = (batch_combined_channel[:, :, 0].real.view(batch_size, num_device, 1) +
                                              batch_combined_channel[:, :, 1:].real @
                                              phase_shift_real.view(batch_size, num_RIS, 1)
                                              - batch_combined_channel[:, :, 1:].imag @
                                              phase_shift_imag.view(batch_size, num_RIS, 1))
            batch_channel_coefficient_imag = (batch_combined_channel[:, :, 0].imag.view(batch_size, num_device, 1) +
                                              batch_combined_channel[:, :, 1:].imag @
                                              phase_shift_real.view(batch_size, num_RIS, 1) +
                                              batch_combined_channel[:, :, 1:].real @
                                              phase_shift_imag.view(batch_size, num_RIS, 1))

            batch_channel_coefficient = torch.sqrt(
                batch_channel_coefficient_real ** 2 + batch_channel_coefficient_imag ** 2).view(batch_size, num_device)
            beamforming = (Net_output[:, -2:])
            c = Net_output[:, :num_device]
            f1, f2 = beamforming[:, 0], beamforming[:, 1]

            c_sum = torch.sum(c, dim=1).unsqueeze(1).cuda()
            mean_class_1_8 = torch.tensor(mean_class).cuda().reshape((1, num_class * num_feature))
            mu_hat = (c_sum @ mean_class_1_8).reshape((batch_size, num_class, num_feature))

            # sigma_hat (64, num_feature)
            sigma_hat = torch.zeros(batch_size, num_feature).cuda()
            temp = c_sum ** 2 @ torch.tensor(var_class).cuda().unsqueeze(0) ** 2

            for i in range(temp.shape[1]):
                sigma_hat[:, i] = (temp[:, i] + sum([c[:, k] ** 2 * feature_noise_var for k in range(num_device)]) +
                                   var_comm_noise / 2 * (f1 ** 2 + f2 ** 2))

            regulazier = torch.zeros((batch_size, num_device)).cuda()
            for k in range(num_device):
                regulazier[:, k] = torch.nn.ReLU()(c[:, k] ** 2 - p_bar[k] * batch_channel_coefficient[:, k] *
                                                   (f1 ** 2 + f2 ** 2) * batch_channel_coefficient[:, k])

            discgain = torch.zeros((batch_size, int(num_class * (num_class - 1) / 2), num_feature)).cuda()

            for class_a in range(num_class):
                for class_b in range(class_a):
                    for idx_m in range(num_feature):
                        discgain[:, int(class_a * (class_a - 1) / 2) + class_b, idx_m] = \
                            ((mu_hat[:, class_a, idx_m] - mu_hat[:, class_b, idx_m]) ** 2 / sigma_hat[:, idx_m])

            loss = (-(2 / num_class * (num_class - 1) * torch.sum(discgain, dim=(1, 2))) +
                    nu * torch.sum(regulazier, dim=1)).mean() * (1 + 0.01 * power_bar)

            total_loss += loss.item()
            print(f'Batch Loss: {loss.item():.4f}')
            loss_history.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    torch.save(GNN.state_dict(), f'./save_model/models/GNN_model_power_{power_bar}.pth')

    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('iters')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()



def evaluate(GNN, test_loader, num_device, power_bar):
    GNN.eval()
    discgain_list = []
    batch_size = test_loader.batch_size
    with torch.no_grad():
        for batch_combined_channel in test_loader:
            batch_combined_channel = batch_combined_channel.cuda()
            input = (batch_combined_channel[:, :, 0].view(batch_size, 1, num_device) +
                     batch_combined_channel[:, :, 1:].sum(-1).view(batch_size, 1, num_device))
            channel_compose = torch.cat((input.real, input.imag), 1).cuda()
            Net_output = GNN.forward(channel_compose).clone()

            c = Net_output[:, :num_device]
            f1, f2 = Net_output[:, -2], Net_output[:, -1]

            c_sum = torch.sum(c, dim=1).unsqueeze(1).cuda()
            mean_class_1_8 = torch.tensor(mean_class).cuda().reshape((1, num_class * num_feature))
            mu_hat = (c_sum @ mean_class_1_8).reshape((batch_size, num_class, num_feature))

            sigma_hat = torch.zeros(batch_size, num_feature).cuda()
            temp = c_sum ** 2 @ torch.tensor(var_class).cuda().unsqueeze(0) ** 2

            for i in range(temp.shape[1]):
                sigma_hat[:, i] = (temp[:, i] + sum([c[:, k] ** 2 * feature_noise_var for k in range(num_device)]) +
                                   var_comm_noise / 2 * (f1 ** 2 + f2 ** 2))

            discgain = torch.zeros((batch_size, int(num_class * (num_class - 1) / 2),
                                    num_feature)).cuda()

            for class_a in range(num_class):
                for class_b in range(class_a):
                    for idx_m in range(num_feature):
                        discgain[:, int(class_a * (class_a - 1) / 2) + class_b, idx_m] = \
                            ((mu_hat[:, class_a, idx_m] - mu_hat[:, class_b, idx_m]) ** 2 / sigma_hat[:, idx_m])

            discriminant_gain = (2 / num_class * (num_class - 1) * torch.sum(discgain, dim=(1, 2))).mean()
                       # * (1 + alpha[power_bar_list.index(power_bar)] * power_bar)
            discgain_list.append(discriminant_gain.item())

    return np.mean(discgain_list)


def main():
    config = get_config_obj()
    power_bar_list = config['params']['power_bar']
    discriminant_gains = []

    train_dataset = CustomDataset(config['data']['power']['train_dir'] + f'/train_combined_channel_power_general.npy')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_dataset = CustomDataset(config['data']['power']['test_dir'] + f'/test_combined_channel_power_general.npy')
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    for power_bar in power_bar_list:
        model_path = f'./save_model/models/GNN_model_power_{power_bar}.pth'
        GNN = Graph_net(2, num_device, num_RIS)

        if os.path.isfile(model_path):
            print(f'Model for power {power_bar} already exists. Skipping training.')
            GNN.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            p_bar = [dbm2pw(power_bar) for _ in range(num_device)]
            nu = 1
            optimizer = optim.SGD(GNN.parameters(), lr=config['training']['learning_rate'])
            train(
                GNN=GNN,
                train_loader=train_loader,
                optimizer=optimizer,
                epochs=config['training']['epochs'],
                num_device=num_device,
                p_bar=p_bar,
                nu=nu,
                power_bar=power_bar
            )
        discriminant_gain = evaluate(GNN, test_loader, num_device, power_bar)
        discriminant_gains.append(discriminant_gain)

    np.save('./save_model/save_results/GNN/discriminant_gains_power.npy', discriminant_gains)
    # plt.plot(power_bar_list, discriminant_gains, marker='o')
    # plt.xlabel('Value of Power bar')
    # plt.ylabel('Discriminant Gain')
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    main()
