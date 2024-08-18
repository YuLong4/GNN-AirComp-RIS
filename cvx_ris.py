import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn import datasets, svm, metrics
import torch
import pickle
from utils.utils import get_config_obj


#load mean, variance, test data, and label
mean_class = np.load('./save_model/save_mean_variance/mean_class_12dim.npy', allow_pickle=True)
var_class = np.load('./save_model/save_mean_variance/var_class_12dim.npy', allow_pickle=True)
data_test_pca_normed = np.load('./save_model/save_mean_variance/data_test_pca_normed_12dim.npy', allow_pickle=True)
label_test = np.load('./save_model/save_mean_variance/label_test_12dim.npy', allow_pickle=True)

# parameters
num_antenna = 1  # number of antenna, N
PCA_dim = 12  # PCA dimension, M
num_device = 3  # number of devices, K
num_class = 4  # number of classes, L
num_ris = 10  ## #number of passive reflecting element equipped by RIS
var_dist_scale = 0.4
var_comm_noise = 1  # communication noise, sigma_{0}^{2}
rng = default_rng()
var_dist = rng.uniform(0, var_dist_scale, (num_device, PCA_dim))  # variance of distortion, delta_{k,m}
power_mdB = 12
power_tx = 10 ** ((power_mdB-30) / 10)  # transmit power, P_{k}
#power_tx = 0.1  # transmit power, P_{k}
bandwidth = 1.5 * 10**6
radius = 500  # BS in the center of circle of radius = 500 m
radius_inner = 450  # Distance between device circle and the server
chl_shadow_std_db = 8  # shadow fading standard deviation = 8 dB
user_dist = (radius - radius_inner) * rng.random((1, num_device)) + radius_inner
user_pl_db = 128.1 + 37.6 * np.log10(user_dist / 1e3)  # path loss in dB
user_pl_db = user_pl_db - chl_shadow_std_db
user_pl = 10 ** (-user_pl_db / 10)
rayli_fading_real_hd = rng.normal(0, 1, (num_antenna, num_device))  # rayleigh fading ~ CN(0,1)
rayli_fading_img_hd = rng.normal(0, 1, (num_antenna, num_device))
rayli_fading_gain_hd = rayli_fading_real_hd ** 2 + rayli_fading_img_hd ** 2
noise_power = 10**(-17.4) * bandwidth  # from You's paper
channel_gain_hd = user_pl * np.ones((num_antenna,1)) * np.sqrt(rayli_fading_gain_hd) / noise_power

ris_dist = (radius - radius_inner) * rng.random(1) + radius_inner
ris_pl_db = 128.1 + 37.6 * np.log10(ris_dist / 1e3)  # path loss in dB
ris_pl_db = ris_pl_db - chl_shadow_std_db
ris_pl = 10 ** (-ris_pl_db / 10)
rayli_fading_real_hr = rng.normal(0, 1, (num_ris, num_device))  # rayleigh fading ~ CN(0,1)
rayli_fading_img_hr = rng.normal(0, 1, (num_ris, num_device))
rayli_fading_gain_hr = rayli_fading_real_hr ** 2 + rayli_fading_img_hr ** 2
channel_gain_hr = ris_pl * np.ones((num_ris, 1)) * np.sqrt(rayli_fading_gain_hr) / noise_power

ris_dist2 = (radius - radius_inner) * rng.random(1) + radius_inner
ris_ap_db = 128.1 + 37.6 * np.log10(ris_dist2 / 1e3)  # path loss in dB
ris_ap_db = ris_ap_db - chl_shadow_std_db
ris_ap = 10 ** (-ris_ap_db / 10)
rayli_fading_real_R = rng.normal(0, 1, (num_antenna , num_ris))  # rayleigh fading ~ CN(0,1)
rayli_fading_img_R = rng.normal(0, 1, (num_antenna, num_ris ))
rayli_fading_gain_R = rayli_fading_real_R ** 2 + rayli_fading_img_R ** 2
channel_gain_R = ris_ap * np.ones((num_antenna, num_ris)) * np.sqrt(rayli_fading_gain_R) / noise_power


############## initialization ###############
f_vec_init = 0.001 * np.ones((num_antenna,1))
v_bar_init = np.array([[1] * num_ris + [0] * num_ris]).T
Y_init=[]
Z_init=[]
p_init = []
q_init = []

for k in range(num_device):
    xk_init = f_vec_init.T @ channel_gain_hd[:,k]
    yk_init = f_vec_init.T @ channel_gain_R @ np.diag(channel_gain_hr[:,k])

    Yk_init1 = np.concatenate((np.real(- yk_init.T @ yk_init),- np.imag(- yk_init.T @ yk_init)), axis = 1)
    Yk_init2 = np.concatenate((np.imag(- yk_init.T @ yk_init), np.real(- yk_init.T @ yk_init)), axis = 1)
    Yk_init = np.concatenate((Yk_init1,Yk_init2),axis = 0)
    zk_init = np.concatenate((np.real((xk_init @ yk_init).T), np.imag((xk_init @ yk_init).T)), axis = 0)

    Y_init += [Yk_init]
    Z_init += [zk_init]
    pk_init = 2 * (Yk_init @ v_bar_init - zk_init)
    qk_init = - v_bar_init.T @ Yk_init @ v_bar_init - (np.abs(xk_init))**2
    p_init += [pk_init]
    q_init += [qk_init]

c_zf_init = np.zeros((num_device,1))

for k in range(num_device):
    channel_gain_init = channel_gain_hd[:,k].reshape(-1,1) + channel_gain_R @ np.diag(channel_gain_hr[:,k]) @ v_bar_init[0:num_ris,]
    c_zf_k = 2 * power_tx * channel_gain_init.T @ f_vec_init * f_vec_init.T @ channel_gain_init
    c_zf_init[k] = c_zf_k

alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))



#two dimension's PCA
##############################################################
def SCA_for_two_PCA(alpha_init,c_zf_init,f_vec_init,v_bar_init,num_antenna,power_tx,channel_gain_hd,channel_gain_hr,channel_gain_R,num_class,num_ris,num_device,idx,var_dist):

    for idx_class_a in range(num_class):
        for idx_class_b in range(idx_class_a):
            for idx_m in  range(2):
                alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] = ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / ((np.sum(var_dist[:, 2 * idx + idx_m].reshape(1,num_device) @ (c_zf_init ** 2)) + var_comm_noise * np.sum(f_vec_init ** 2)) / ((np.sum(c_zf_init)) ** 2) + var_class[2 * idx + idx_m])

    disc_init = 2 * alpha_init.sum() / num_class / (num_class-1)

    last_value = disc_init
    diff = last_value
    count = 1

    while(diff>1e-2 and count<100):
        c_zf = cp.Variable((num_device, 1))
        v_bar = cp.Variable((2 * num_ris))

        p = []
        q = []
        x = []
        Y = []
        Z = []
        for k in range(num_device):
            xk = f_vec_init.T @ channel_gain_hd[:,k]
            yk = f_vec_init.T @ channel_gain_R @ np.diag(channel_gain_hr[:,k])
            Yk = np.bmat([[(- yk.T @ yk), np.zeros((num_ris,num_ris))],[np.zeros((num_ris,num_ris)), (- yk.T @ yk)]])

            Zk = np.hstack([(xk @ yk), np.zeros((num_ris))])
            x += [xk]
            Y += [Yk]
            Z += [Zk]
            pk = 2 * (Yk @ v_bar - Zk)
            qk = - v_bar.T @ Yk @ v_bar - (np.abs(xk))**2
            p += [pk]
            q += [qk]



        constrains = []
        alpha = {}
        for idx_m in range(2):
            alpha[idx_m] = cp.Variable((int(num_class * (num_class-1) / 2), 1))

        for idx_theta in range(num_ris):
            constrains += [(v_bar[idx_theta]) **2 + (v_bar[idx_theta + num_ris]) **2 <= 1]

        for idx_device in range(num_device):
            #constrains += [(c_zf[idx_device] ** 2 + 2 * power_tx *((p[idx_device]).T @ v_bar + q[idx_device])) <= 0]
            constrains += [ c_zf[idx_device] ** 2 + (2 * Y[idx_device] @ v_bar_init - 2 * (Z[idx_device].reshape(-1,1))).T @ (v_bar - v_bar_init.reshape(-1))  <= 0]


        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    constrains += [(cp.sum_squares(cp.multiply(c_zf, np.sqrt(var_dist[:, 2 * idx + idx_m].reshape(-1, 1))))
                    + var_comm_noise * np.sum(f_vec_init ** 2))
                    + (cp.sum(c_zf)) ** 2 * var_class[2 * idx + idx_m]
                    - np.sum(c_zf_init) ** 2 * ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2 / alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) - 2 * (mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2 * np.sum(c_zf_init) / alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] * cp.sum(c_zf - c_zf_init) + (np.sum(c_zf_init) * (mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) / alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) ** 2 * (alpha[idx_m][int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b] - alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) <= 0]

        for idx_device in range(num_device):
            constrains += [c_zf[idx_device] >= 0]


        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    constrains += [alpha[idx_m][int(idx_class_a * (idx_class_a - 1) / 2) + idx_class_b] >= 0]


        objective = cp.Maximize(2 * cp.sum(alpha[0] + alpha[1]) / num_class / (num_class-1))
        prob = cp.Problem(objective, constrains)
        prob.solve(solver=cp.ECOS, verbose=False)

        stepsize = 0.1

        c_zf_init = c_zf.value * stepsize + c_zf_init * (1-stepsize)
        v_bar_init = v_bar.value * stepsize + v_bar_init.reshape(-1) * (1-stepsize)
        temp1 = alpha[0].value * stepsize
        alpha_init[:, 0] = temp1.reshape(int(num_class * (num_class-1) / 2)) + alpha_init[:, 0] * (1-stepsize)
        temp2 = alpha[1].value * stepsize
        alpha_init[:, 1] = temp2.reshape(int(num_class * (num_class-1) / 2)) + alpha_init[:, 1] * (1-stepsize)
        # if alpha_init[:, 0] < 0 :
        #     alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))
        # if alpha_init[:, 1] < 0 :
        #     alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))


        diff = abs(prob.value - last_value)
        last_value = prob.value
        print("The discriminant_1 gain after {}-th interation is: {}, diff:{}".format(count, last_value,diff))
        count += 1

    #print("The discriminant gain after optimization is: {}".format(prob.value))
    return c_zf_init,v_bar_init,last_value,alpha_init


#################### Second round SCA #######################
def SCA_c_Rf(alpha_init,c_zf,v_bar,f_vec_init,num_antenna,power_tx,channel_gain_hd,channel_gain_hr,channel_gain_R,num_class,num_ris,num_device,idx,var_dist,c_zf_init):

    f_vec = cp.Variable((num_antenna))
    alpha = {}
    for idx_m in range(2):
        alpha[idx_m] = cp.Variable(((int(num_class * (num_class-1) / 2), 1)))

    # alpha_init = 1e-2 * np.ones(( int (num_class * (num_class-1) / 2), 2))
    for idx_class_a in range(num_class):
        for idx_class_b in range(idx_class_a):
            for idx_m in  range(2):
                alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] = ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / ((np.sum(var_dist[:, 2 * idx + idx_m].reshape(1,num_device) @ (c_zf_init ** 2)) + var_comm_noise * np.sum(f_vec_init ** 2)) / ((np.sum(c_zf_init)) ** 2) + var_class[2 * idx + idx_m])


    disc_init = 2 * alpha_init.sum() / num_class / (num_class-1)
    last_value = disc_init
    diff = last_value
    count = 1

    objective=[]
    constrains=[]

    channel_gain_device = []
    for k in range(num_device):
        channel_gain_k = channel_gain_hd[:,k] + channel_gain_R @ np.diag(channel_gain_hr[:,k]) @ v_bar[0:num_ris]
        channel_gain_device.append(channel_gain_k)

    while(diff>1e-2 and count<100):
        for idx_device in range(num_device):
            constrains += [2 * power_tx * channel_gain_device[idx_device] @ f_vec_init.reshape(-1) * f_vec_init.reshape(-1).T @ channel_gain_device[idx_device].T
                + 4 * power_tx * channel_gain_device[idx_device] @ channel_gain_device[idx_device].T * f_vec_init.reshape(-1) @ (f_vec - f_vec_init.reshape(-1)) - c_zf[idx_device] **2 >= 0]


        for idx_antenna in range(num_antenna):
            constrains += [f_vec[idx_antenna] >= 0]

        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    constrains += [np.sum(np.multiply(c_zf, np.sqrt(var_dist[:, 2 * idx + idx_m].reshape(-1, 1))) ** 2)
                    + var_comm_noise * cp.sum_squares(f_vec)
                    + (np.sum(c_zf)) ** 2 * var_class[2 * idx + idx_m]
                    - np.sum(c_zf) ** 2 * ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / (alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m])
                    + np.sum(c_zf) ** 2 * ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / (alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) **2 *  (alpha[idx_m][int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b] -  alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m])  <= 0]

        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    constrains += [alpha[idx_m][int(idx_class_a * (idx_class_a - 1) / 2) + idx_class_b] >= 0]

        objective = cp.Maximize(2 * cp.sum(alpha[0] + alpha[1]) / num_class / (num_class-1))
        prob = cp.Problem(objective, constrains)
        prob.solve(solver=cp.ECOS, verbose=False)

        stepsize = 0.1
        f_vec_init = f_vec.value.reshape((-1,1)) * stepsize + f_vec_init * (1-stepsize)
        diff = last_value
        temp1 = alpha[0].value * stepsize
        alpha_init[:, 0] = temp1.reshape(int(num_class * (num_class-1) / 2)) + alpha_init[:, 0] * (1-stepsize)
        temp2 = alpha[1].value * stepsize
        alpha_init[:, 1] = temp2.reshape(int(num_class * (num_class-1) / 2)) + alpha_init[:, 1] * (1-stepsize)
        diff = abs(prob.value - last_value)
        last_value = prob.value

        print("The discriminant_2 gain after {}-th interation is: {}, diff:{}".format(count, last_value,diff))
        count += 1

    return c_zf,f_vec_init,alpha_init,last_value


def add_noise_to_normed_pca(data_test_pca_normed, c_zf_init, f_vec_init):
    var_dist = rng.uniform(0, var_dist_scale, (num_device, 2))
    data_test_pca_add_noise = (np.sum(c_zf_init) * data_test_pca_normed + c_zf_init.T @ var_dist + np.sum(f_vec_init * var_comm_noise)) / np.sum(c_zf_init)
    return data_test_pca_add_noise


def model_inference(data, label, model):
    predicted = model.predict(data)
    accuracy = metrics.accuracy_score(label, predicted)
    return accuracy


def compute_discriminant_gain(c, f):
    batch_size = config['training']['batch_size']
    c = np.squeeze(c, axis=-1)
    f = np.squeeze(f, axis=-1)
    c = np.tile(c, (batch_size, 1))
    f = np.tile(f, (batch_size,))
    c = torch.from_numpy(c).cuda()
    f = torch.from_numpy(f).cuda()
    c_sum = torch.sum(c, dim=1).unsqueeze(1).cuda()
    mean_class_1_8 = torch.tensor(mean_class).cuda().reshape((1, num_class * PCA_dim))
    mu_hat = (c_sum @ mean_class_1_8).reshape((batch_size, num_class, PCA_dim))

    sigma_hat = torch.zeros(batch_size, PCA_dim).cuda()
    temp = c_sum ** 2 @ torch.tensor(var_class).cuda().unsqueeze(0) ** 2

    for i in range(temp.shape[1]):
        sigma_hat[:, i] = (temp[:, i] + sum([c[:, k] ** 2 * var_dist_scale for k in range(num_device)]) +
                           var_comm_noise / 2 * (f ** 2))

    discgain = torch.zeros((batch_size, int(num_class * (num_class - 1) / 2),
                            PCA_dim)).cuda()

    for class_a in range(num_class):
        for class_b in range(class_a):
            for idx_m in range(PCA_dim):
                discgain[:, int(class_a * (class_a - 1) / 2) + class_b, idx_m] = \
                    ((mu_hat[:, class_a, idx_m] - mu_hat[:, class_b, idx_m]) ** 2 / sigma_hat[:, idx_m])

    discriminant_gain = (2 / num_class * (num_class - 1) * torch.sum(discgain, dim=(1, 2))).mean()
    return discriminant_gain


config = get_config_obj()

num_ris_list = np.array(config['params']['num_RIS'])

# #####the below is computating the accuracy with the change of number sizes.#####
discriminant_gain_list = np.zeros((len(num_ris_list), 1))
discriminant_gain_baseline_list = np.zeros((len(num_ris_list), 1))
discriminant_gain_random_list = np.zeros((len(num_ris_list), 1))

for i in range(len(num_ris_list)):
    num_ris = int(num_ris_list[i])
    var_dist = rng.uniform(0, var_dist_scale, (num_device, PCA_dim))  # variance of distortion, delta_{k,m}

    user_dist = (radius - radius_inner) * rng.random((1, num_device)) + radius_inner
    user_pl_db = 128.1 + 37.6 * np.log10(user_dist / 1e3)  # path loss in dB
    user_pl_db = user_pl_db - chl_shadow_std_db
    user_pl = 10 ** (-user_pl_db / 10)

    # hr
    ris_dist = (radius - radius_inner) * rng.random(1) + radius_inner
    ris_pl_db = 128.1 + 37.6 * np.log10(ris_dist / 1e3)  # path loss in dB
    ris_pl_db = ris_pl_db - chl_shadow_std_db
    ris_pl = 10 ** (-ris_pl_db / 10)
    rayli_fading_real_hr = rng.normal(0, 1, (num_ris,1))  # rayleigh fading ~ CN(0,1)
    rng3 = default_rng()
    rayli_fading_img_hr = rng3.normal(0, 1, (num_ris,1))
    for j in range(1,num_device):
        rng2 = default_rng()
        rng3 = default_rng()
        rayli_fading_real_hr = np.hstack((rayli_fading_real_hr,rng2.normal(0, 1, (num_ris, 1))))   # rayleigh fading ~ CN(0,1)
        rayli_fading_img_hr = np.hstack((rayli_fading_img_hr,rng3.normal(0, 1, (num_ris, 1))))
    rayli_fading_gain_hr = rayli_fading_real_hr ** 2 + rayli_fading_img_hr ** 2
    noise_power = 10 ** (-17.4) * bandwidth  # from You's paper
    # channel_gain = user_pl * np.ones((1, num_antenna)) * np.sqrt(rayli_fading_gain) / noise_power
    channel_gain_hr = ris_pl * np.ones((num_ris,1)) * np.sqrt(rayli_fading_gain_hr) / noise_power

    ris_dist2 = (radius - radius_inner) * rng.random(1) + radius_inner
    ris_ap_db = 128.1 + 37.6 * np.log10(ris_dist2 / 1e3)  # path loss in dB
    ris_ap_db = ris_ap_db - chl_shadow_std_db
    ris_ap = 10 ** (-ris_ap_db / 10)
    rayli_fading_real_R = rng.normal(0, 1, (num_antenna, num_ris))  # rayleigh fading ~ CN(0,1)
    rayli_fading_img_R = rng.normal(0, 1, (num_antenna, num_ris))
    rayli_fading_gain_R = rayli_fading_real_R ** 2 + rayli_fading_img_R ** 2
    channel_gain_R = ris_ap * np.ones((num_antenna, num_ris)) * np.sqrt(rayli_fading_gain_R) / noise_power

    power_mdB = 12
    power_tx = 10 ** ((power_mdB - 30) / 10)
    data_test_pca_add_noise = np.ones((np.size(data_test_pca_normed,0),PCA_dim))
    #baseline
    data_test_pca_add_noise_baseline = np.ones((np.size(data_test_pca_normed, 0), PCA_dim))
    discriminant_gain_init = []


    # for idx in range(0,int(PCA_dim/2)):
    #     ############## initialization ###############
    #     f_vec_init = np.ones((num_antenna,1)) / (np.sqrt(num_antenna * 2 * 10 / power_tx))
    #     v_bar_init = np.array([[1] * num_ris + [0] * num_ris]).T
    #     Y_init=[]
    #     Z_init=[]
    #     p_init = []
    #     q_init = []
    # 
    #     for k in range(num_device):
    #         xk_init = f_vec_init.T @ channel_gain_hd[:,k]
    #         yk_init = f_vec_init.T @ channel_gain_R @ np.diag(channel_gain_hr[:,k])
    #         Yk_init1 = np.concatenate((np.real(- yk_init.T @ yk_init),- np.imag(- yk_init.T @ yk_init)), axis = 1)
    #         Yk_init2 = np.concatenate((np.imag(- yk_init.T @ yk_init), np.real(- yk_init.T @ yk_init)), axis = 1)
    #         Yk_init = np.concatenate((Yk_init1,Yk_init2),axis = 0)
    #         zk_init = np.concatenate((np.real((xk_init @ yk_init).T), np.imag((xk_init @ yk_init).T)), axis = 0)
    #         Y_init += [Yk_init]
    #         Z_init += [zk_init]
    #         pk_init = 2 * (Yk_init @ v_bar_init - zk_init)
    #         qk_init = - v_bar_init.T @ Yk_init @ v_bar_init - (np.abs(xk_init))**2
    #         p_init += [pk_init]
    #         q_init += [qk_init]
    #     c_zf_init = np.zeros((num_device,1))
    # 
    #     for k in range(num_device):
    #         channel_gain_init = channel_gain_hd[:,k].reshape(-1,1) + channel_gain_R @ np.diag(channel_gain_hr[:,k]) @ v_bar_init[0:num_ris,]
    #         c_zf_k = 2 * power_tx * channel_gain_init.T @ f_vec_init * f_vec_init.T @ channel_gain_init
    #         c_zf_init[k] = c_zf_k
    #     alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))
    # 
    #     count = 1
    #     disc_init = 2 * alpha_init.sum() / num_class / (num_class-1)
    #     last_value = disc_init
    #     diff = 1
    #     discri = 0
    # 
    #     while (diff > 1e-2 and count <100):
    #         c_zf, v_bar, discri, alpha_init = SCA_for_two_PCA(alpha_init,c_zf_init,f_vec_init,v_bar_init,num_antenna,power_tx,channel_gain_hd,channel_gain_hr,channel_gain_R,num_class,num_ris,num_device, idx, var_dist)
    #         c_zf_init,f_vec_init,alpha_init,discri = SCA_c_Rf(alpha_init,c_zf,v_bar,f_vec_init,num_antenna,power_tx,channel_gain_hd,channel_gain_hr,channel_gain_R,num_class,num_ris,num_device, idx, var_dist,c_zf_init)
    #         v_bar_init = v_bar
    #         diff = abs(discri - last_value)
    #         last_value = discri
    #         count += 1
    #     print ("while finish {}-th".format(idx))
    #     discriminant_gain_init.append(discri)
    #     data_test_pca_add_noise[:,idx*2:idx*2+2] = add_noise_to_normed_pca(data_test_pca_normed[:,idx*2:idx*2+2], c_zf_init, f_vec_init)
    # 
    # print(f'c_zf_init: {c_zf_init}  f_vec_init: {f_vec_init}')
    # discriminant_gain_list[i] = compute_discriminant_gain(c=c_zf_init, f=f_vec_init).cpu()
    # # discriminant_gain_list[i] = np.sum(discriminant_gain_init)

    # without RIS
    f_vec_init = np.ones((num_antenna, 1))  # beamforming init, f 改成0.001*
    c_zf_init = np.zeros((num_device,1))
    v_bar_init = rng.random(num_ris*2).T
    for idx in range (num_ris):
        v_bar_init[idx+num_ris] = (np.sqrt(1 - (v_bar_init[idx]) **2 ))

    for k in range(num_device):
        channel_gain_init = channel_gain_hd[:,k].reshape(-1,1) + channel_gain_R @ np.diag(channel_gain_hr[:,k]) @ (v_bar_init[0:num_ris,]).reshape(-1,1)
        c_zf_k = 2 * power_tx * channel_gain_init.T @ f_vec_init * f_vec_init.T @ channel_gain_init
        c_zf_init[k] = 1e-4 * c_zf_k
    alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))

    disc_init = 0
    for idx in range(0, int(PCA_dim / 2)):
        alpha_init = np.zeros((int(num_class * (num_class - 1) / 2), 2))
        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] = ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / ((np.sum(var_dist[:, 2 * idx + idx_m].reshape(1,num_device) @ (c_zf_init ** 2)) + var_comm_noise * np.sum(f_vec_init ** 2)) / ((np.sum(c_zf_init)) ** 2) + var_class[2 * idx + idx_m])


        disc_init += 2 * alpha_init.sum() / num_class / (num_class-1)
        data_test_pca_add_noise_baseline[:, idx * 2:idx * 2 + 2] = add_noise_to_normed_pca(data_test_pca_normed[:, idx * 2:idx * 2 + 2], c_zf_init, f_vec_init)

    print(f'c_zf_init: {c_zf_init}  f_vec_init: {f_vec_init}')
    discriminant_gain_baseline_list[i] = compute_discriminant_gain(c=c_zf_init, f=f_vec_init).cpu()
    
    # Random
    f_vec_init = np.zeros((num_antenna, 1))
    c_zf_init = np.zeros((num_device, 1))

    for _ in range(100):
        f_vec_init += rng.random((num_antenna, 1)) * 0.6
        c_zf_init += rng.random((num_device, 1))

    f_vec_init /= 100
    c_zf_init /= 100
    # v_bar_init = rng.random(num_ris * 2).T
    # for idx in range(num_ris):
    #     v_bar_init[idx + num_ris] = (np.sqrt(1 - (v_bar_init[idx]) ** 2))  # 后半部分为虚部
    # 
    # for k in range(num_device):
    #     channel_gain_init = channel_gain_hd[:,k].reshape(-1,1) + channel_gain_R @ np.diag(channel_gain_hr[:,k]) @ (v_bar_init[0:num_ris,]).reshape(-1,1)
    #     c_zf_k = 2 * power_tx * channel_gain_init.T @ f_vec_init * f_vec_init.T @ channel_gain_init
    #     c_zf_init[k] = 1e-4 * c_zf_k
    # alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))
    # 
    # disc_init = 0
    # for idx in range(0, int(PCA_dim / 2)):
    #     alpha_init = np.zeros((int(num_class * (num_class - 1) / 2), 2))
    #     for idx_class_a in range(num_class):
    #         for idx_class_b in range(idx_class_a):
    #             for idx_m in range(2):
    #                 alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] = ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / ((np.sum(var_dist[:, 2 * idx + idx_m].reshape(1,num_device) @ (c_zf_init ** 2)) + var_comm_noise * np.sum(f_vec_init ** 2)) / ((np.sum(c_zf_init)) ** 2) + var_class[2 * idx + idx_m])
    # 
    #     disc_init += 2 * alpha_init.sum() / num_class / (num_class-1)
    #     data_test_pca_add_noise_baseline[:, idx * 2:idx * 2 + 2] = add_noise_to_normed_pca(data_test_pca_normed[:, idx * 2:idx * 2 + 2], c_zf_init, f_vec_init)

    discriminant_gain_random_list[i] = compute_discriminant_gain(c=c_zf_init, f=f_vec_init).cpu()

# np.save('./save_model/save_results/discriminant_gain_ris_with_ris.npy', discriminant_gain_list)

# without RIS
np.save('./save_model/save_results/discriminant_gain_ris_without_ris.npy', discriminant_gain_baseline_list)

# Random
# np.save('./save_model/save_results/discriminant_gain_ris_random.npy', discriminant_gain_random_list)

print('save done')

def generate_channel_gain(num_sample):
    for num_ris in num_ris_list:
        channel_gain_hd_sample = []
        channel_gain_hr_sample = []
        channel_gain_R_sample = []
        print(f'generating channels of ris {num_ris}')
        for count in range(num_sample):
            var_dist = rng.uniform(0, var_dist_scale, (num_device, PCA_dim))  # variance of distortion, delta_{k,m}
            user_dist = (radius - radius_inner) * rng.random((1, num_device)) + radius_inner
            user_pl_db = 128.1 + 37.6 * np.log10(user_dist / 1e3)  # path loss in dB
            user_pl_db = user_pl_db - chl_shadow_std_db
            user_pl = 10 ** (-user_pl_db / 10)

            # hd
            # rayli_fading_real = rng.normal(0, 1, (num_device, num_antenna))  # rayleigh fading ~ CN(0,1)
            # rayli_fading_img = rng.normal(0, 1, (num_device, num_antenna))
            rayli_fading_real_hd = rng.normal(0, 1, (num_antenna,1))  # rayleigh fading ~ CN(0,1)
            rng3 = default_rng()
            rayli_fading_img_hd = rng3.normal(0, 1, (num_antenna,1))
            for j in range(1,num_device):
                rng2 = default_rng()
                rng3 = default_rng()
                rayli_fading_real_hd = np.hstack((rayli_fading_real_hd,rng2.normal(0, 1, (num_antenna, 1))))   # rayleigh fading ~ CN(0,1)
                rayli_fading_img_hd = np.hstack((rayli_fading_img_hd,rng3.normal(0, 1, (num_antenna, 1))))

            rayli_fading_gain_hd = rayli_fading_real_hd ** 2 + rayli_fading_img_hd ** 2
            noise_power = 10 ** (-17.4) * bandwidth  # from You's paper
            # channel_gain = user_pl * np.ones((1, num_antenna)) * np.sqrt(rayli_fading_gain) / noise_power
            channel_gain_hd = user_pl * np.ones((num_antenna,1)) * np.sqrt(rayli_fading_gain_hd) / noise_power

            # hr
            ris_dist = (radius - radius_inner) * rng.random(1) + radius_inner
            ris_pl_db = 128.1 + 37.6 * np.log10(ris_dist / 1e3)  # path loss in dB
            ris_pl_db = ris_pl_db - chl_shadow_std_db
            ris_pl = 10 ** (-ris_pl_db / 10)
            rayli_fading_real_hr = rng.normal(0, 1, (num_ris,1))  # rayleigh fading ~ CN(0,1)
            rng3 = default_rng()
            rayli_fading_img_hr = rng3.normal(0, 1, (num_ris,1))
            for j in range(1,num_device):
                rng2 = default_rng()
                rng3 = default_rng()
                rayli_fading_real_hr = np.hstack((rayli_fading_real_hr,rng2.normal(0, 1, (num_ris, 1))))   # rayleigh fading ~ CN(0,1)
                rayli_fading_img_hr = np.hstack((rayli_fading_img_hr,rng3.normal(0, 1, (num_ris, 1))))

            rayli_fading_gain_hr = rayli_fading_real_hr ** 2 + rayli_fading_img_hr ** 2
            noise_power = 10 ** (-17.4) * bandwidth  # from You's paper
            # channel_gain = user_pl * np.ones((1, num_antenna)) * np.sqrt(rayli_fading_gain) / noise_power
            channel_gain_hr = ris_pl * np.ones((num_ris,1)) * np.sqrt(rayli_fading_gain_hr) / noise_power

            ris_dist2 = (radius - radius_inner) * rng.random(1) + radius_inner
            ris_ap_db = 128.1 + 37.6 * np.log10(ris_dist2 / 1e3)  # path loss in dB
            ris_ap_db = ris_ap_db - chl_shadow_std_db
            ris_ap = 10 ** (-ris_ap_db / 10)
            rayli_fading_real_R = rng.normal(0, 1, (num_antenna, num_ris))  # rayleigh fading ~ CN(0,1)
            rayli_fading_img_R = rng.normal(0, 1, (num_antenna, num_ris))
            rayli_fading_gain_R = rayli_fading_real_R ** 2 + rayli_fading_img_R ** 2
            channel_gain_R = ris_ap * np.ones((num_antenna, num_ris)) * np.sqrt(rayli_fading_gain_R) / noise_power

            channel_gain_hd_sample.append(channel_gain_hd)
            channel_gain_hr_sample.append(channel_gain_hr)
            channel_gain_R_sample.append(channel_gain_R)
        np.savez(f'./data/from_cvx/ris/channels/channels_ris_{num_ris}.npz',
                array1=channel_gain_hd_sample, array2=channel_gain_hr_sample, array3=channel_gain_R_sample)


if __name__ == '__main__':
    # generate_channel_gain(num_sample=10000)
    pass