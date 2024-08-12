### 8/12
#### Issue: 
不同p_bar训练出的模型最终收敛到相同的loss值，计算出的discriminant gain的值也基本相同
#### Findings: 
需要修改loss函数，让p_bar对loss的影响增大