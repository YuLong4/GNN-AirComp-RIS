### 8/12
#### Issue: 
不同p_bar训练出的模型最终收敛到相同的loss值，计算出的discriminant gain的值也基本相同
#### Findings: 
需要修改loss函数，让p_bar对loss的影响增大</br>
++ num_RIS是同样的情况，问题都出在取值对loss和gain没有什么影响。因为存在对num_device求和的操作，所以num_device对loss和gain的变化明显