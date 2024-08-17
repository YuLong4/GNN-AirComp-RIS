### 8/17
#### update
the path of device channel gain from cvx method is ./data/from_cvx/device/channels
the path of power channel gain from cvx method is ./data/from_cvx/power/channels
device的channel gain按设备数不同而不同，power的channel gain与power值无关
从cvx里生成的channel gain形式为由channel_gain_hd, channel_gain_R, channel_gain_hr组成的npz file
然后在GNN代码中生成channels部分进行替换
### 8/16
#### update
使用较小的power_bar: [0, 2, 4, 6, 8, 10, 12]</br>
使用cvx的channel_gain(将channel_gain_hd, channel_gain_R, channel_gain_hr保存起来给GNN训练用)</br>
由cvx代码生成10000份数据，分为8000份训练、2000份测试
### 8/15
#### update
```text
common config:
num_antenna = 1
num_RIS = 10
num_device = 3
num_class = 4
num_feature = 12
power_bar = [0, 2, 4, 6, 8, 10, 12]

CVX_based_without_RIS: v值随机
Random: c、v、f都随机
```
图在drafts.ipynb中

### 8/12
#### Issue: 
不同p_bar训练出的模型最终收敛到相同的loss值，计算出的discriminant gain的值也基本相同
#### Findings: 
需要修改loss函数，让p_bar对loss的影响增大</br>
++ num_RIS是同样的情况，问题都出在取值对loss和gain没有什么影响。因为存在对num_device求和的操作，所以num_device对loss和gain的变化明显
#### Next action
需要修改loss和gain的形式使得num_RIS和p_bar对值有明显影响
#### Resolved
因为论文原图也是不变化的，所以保持现状
