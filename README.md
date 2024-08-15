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

### 8/15
#### update
```text
common config:
num_antenna = 1
num_RIS = 10
num_device = 10
num_class = 4
num_feature = 12
power_bar = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

CVX_based_without_RIS: v值随机
Random: c、v、f都随机
```
