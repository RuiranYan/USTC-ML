# MLlab2 report

## 实验要求

### 实验简介

通过第八章集成学习的课程学习，实现一个自己的xgboost。具体的任务是以一个决策树为基模型的xgboost实现一个回归任务。

### 数据集介绍

回归模型的训练数据集如文件夹里面给定，包含7154行，41列，前40列是feature，最后一列是要预测的标签。可以自行在训练数据里面划分出部分数据当作验证集合。

### 提交要求



## 实验原理

XGBoost 是由多个基模型组成的一个加法模型，假设第k个基本模型是f_k(x), 那么前t个模型组成的模型的输出为
$$
\widehat{y}_{i}^{(t)}=\sum_{k=1}^{t} f_{k}\left(x_{i}\right)=\widehat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)
$$
其中$x_i$为第表示第i个训练样本，$y_i$表示第i个样本的真实标签； $y ̂_i^{(t)}$表示前t个模型对第i个样本的标签最终预测值。

在学习第t个基模型时，XGBoost要优化的目标函数:
$$
\begin{aligned}
\operatorname{Obj}^{(t)} &=\sum_{i=1}^{n} \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t)}\right)+\sum_{k=1}^{t} \text { penalt } y\left(f_{k}\right) \\
&=\sum_{i=1}^{n} \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)\right)+\sum_{k=1}^{t} \text { penalt } y\left(f_{k}\right) \\
&=\sum_{i=1}^{n} \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)\right)+\text { penalt } y\left(f_{t}\right)+\text { constant }
\end{aligned}
$$
其中n表示训练样本的数量, $penalty(f_k)$表示对第k个模型的复杂度的惩罚项，由于依次学习每个基模型，所以当学习第t个基模型时，前t−1个基模型是固定的，其penalty是常数。。$\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t)}\right)$表示损失函数。

例如二分类问题中，损失函数为
$$
\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t)}\right)=-y_{i} \cdot \log p\left(\widehat{y}_{i}^{(t)}=1 \mid x_{i}\right)-\left(1-y_{i}\right) \cdot \log \left(1-p\left(\widehat{y}_{i}^{(t)}=1 \mid x_{i}\right)\right)
$$
回归问题中损失函数为
$$
\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t)}\right)=\left(y_{i}-\widehat{y}_{i}^{(t)}\right)^{2}
$$
学习第t个基模型时，要优化的目标是：
$$
\operatorname{Obj}^{(t)} =\sum_{i=1}^{n} \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)\right)+\text { penalt } y\left(f_{t}\right)+\text { constant }
$$
将$\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)\right)$在$\widehat{y}_{i}^{(t-1)}$处泰勒展开可得：
$$
\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)\right) \approx \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right)+g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)
$$
其中
$$
g_{i}=\frac{\partial \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right)}{\partial \widehat{y}_{i}^{(t-1)}}, h_{i}=\frac{\partial^{2} \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right)}{\partial\left(\bar{y}_{i}^{(t-1)}\right)^{2}}
$$
此时$\operatorname{Obj}^{(t)}$去掉常数项$\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right)$和$\operatorname{constant}$可得目标函数为
$$
O b j^{(t)}=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\text { penalt } y\left(f_{t}\right)
$$
实验要解决回归问题，于是
$$
\operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right) = \left(y_{i}- \widehat{y}_{i}^{(t-1)}\right)^2
$$
故
$$
g_{i}=\frac{\partial \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right)}{\partial \widehat{y}_{i}^{(t-1)}}=-2\left(y_{i}- \widehat{y}_{i}^{(t-1)}\right),
h_{i}=\frac{\partial^{2} \operatorname{loss}\left(y_{i}, \widehat{y}_{i}^{(t-1)}\right)}{\partial\left(\widehat{y}_{i}^{(t-1)}\right)^{2}}=2
$$
本次实验使用决策树作为基模型。假设决策树有T个叶子节点，每个叶子节点对应有一个权重。决策树模型就是将输入$x_i$映射到某个叶子节点，决策树模型的输出就是这个叶子节点的权重。

即$f(x_i)=w_{q(x_i)}$，w是一个要学的T维的向量其中$q(x_i)$表示把输入$x_i$映射到的叶子节点的索引。例如：$q(x_i)=3$，那么模型输出第三个叶子节点的权重，即$f(x_i)=w_3$。

决策树模型中惩罚项为：
$$
\text { penalt } y(f)=\gamma \cdot T+\frac{1}{2} \lambda \cdot\|w\|^{2}
$$
其中$T$为叶子节点的数目，$\gamma,\lambda$为可调的超参。

当树结构确定时，用$I_j$表示分配到第$j$个叶子节点上的样本，由前面可知：
$$
\begin{aligned}
O b j^{(t)}=& \sum_{i=1}^{n}\left[g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\text { penalt } y\left(f_{t}\right) \\
=& \sum_{i=1}^{n}\left[g_{i} w_{q\left(x_{i}\right)}+\frac{1}{2} h_{i} w_{q\left(x_{i}\right)}^{2}\right]+\gamma \cdot T+\frac{1}{2} \lambda \cdot\|w\|^{2} \\
=& \sum_{j=1}^{T}\left[\left(\sum_{x \in I_{j}} g_{i}\right) \cdot w_{j}+\frac{1}{2} \cdot\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) \cdot w_{j}^{2}\right]+\gamma \cdot T
\end{aligned}
$$
记：
$$
G_{j}=\sum_{i \in I_{j}} g_{i}, \quad H_{j}=\sum_{i \in I_{j}} h_{i}
$$
则目标函数变为：
$$
O b j^{(t)}=\sum_{j=1}^{T}\left[G_{j} w_{j}+\frac{1}{2}\left(H_{j}+\lambda\right) w_{j}^{2}\right]+\gamma T
$$
该式为二次函数，能求解出解析解$w^*_j = -\frac{G_j}{H_j+\lambda}$，得出$O b j^{(t)}$的极值：
$$
O b j^{(t)} = -\frac{1}{2} \sum_{j=1}^{T}\frac{G^2_j}{H_j+\lambda} +\gamma T
$$
考虑每次将节点划分为左孩子和右孩子，划分前：
$$
Obj_1 = -\frac{1}{2} \frac{G^2}{H+\lambda}+\gamma
$$
划分后：
$$
Obj_2 = -\frac{1}{2} (\frac{G^2_L}{H_L+\lambda}+\frac{G^2_R}{H_R+\lambda})+2 \gamma
$$
要让划分后$Obj$减少最多，可令$gain = Obj_1-Obj_2$，令$gain$最大即可。

注意到这里有恒等关系$G=G_L+G_R, H=H_L+H_R$，这在后面简化算法有重要意义。

找最大收益划分的算法使用贪心算法，，即对每个特征按该特征排序后用每个值划分直到找到最大的gain，返回最大gain对应的特征和划分值。

通过贪心算法得到第M颗树后，更新最终预测结果为:$y ̂_i=\sum_{t=1}^{M}f_t(x_i)$，知道更新到设定树上限为止。

最终得到的$\widehat{y}_i$即为所求。

实验的评价指标可以考虑如下指标：

* RMSE指标，其值越小越好
* R^2指标，其值越大越好

## 决策树停止划分的标准

本次实验采取的决策是停止划分的标准是简单的限制最大深度，即在实例化模型的过程中要给出参数max_depth，这个参数将限制每颗树的最大深度，当决策树划分到该深度时结束划分。

## 代码讲解





## 实验中遇到的问题及解决方案

**遇到的问题1：**

时间开销过大，生成决策树时间过长。

**解决方案**



**遇到的问题2：**

决策树鲁棒性问题

**解决方案**



## 实验结果展示
