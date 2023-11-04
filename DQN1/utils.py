import torch
import torch.nn as nn
class Memory():
    #memory流程为：
    #采集阶段：添加sample
    #学习阶段：读取sample调整网络，读取buffer调整权重，更新权重赋值
    #初始化
    def __init__(self,size):
        self.size=size
        self.replay_buffer=[]
        self.buffer_bias=[]
    #添加样本
    def add_sample(self,sample,bias):
        self.replay_buffer.append(sample)
        self.buffer_bias.append(bias)
        if len(self.replay_buffer)>self.size:
            self.replay_buffer.pop(0)
            self.buffer_bias.pop(0)
    #优先经验回放
    def sample(self,batch_size):
        if(len(self.replay_buffer)<batch_size):
            return None
        else:
            prob=torch.softmax(torch.log(self.buffer_bias),dim=0)
            batch_size=min(batch_size,len(self.replay_buffer))
            index=torch.multinomial(prob,batch_size,replacement=True)
            out=[]
            for i in index:
                out.append(self.replay_buffer[i])
            samples=list(zip(*out))
            return samples
    #如果是off policy策略无关的值迭代，那么之前的经验还是可以用的，
    #不过需要更新经验权重，也就是用当前动作价值函数重新修正权重
    def bias_renew(self,biass):
        self.buffer_bias=biass
    def get_buffer(self):
        temp=self.replay_buffer
        #返回四元组列表用于修正bias
        return list(zip(*temp))
    def memory_size(self):
        return len(self.replay_buffer)


class Q_network(nn.Module):
    def __init__(self,state_size,action_size):
        super(Q_network,self).__init__()
        self.fc1=nn.Linear(state_size,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,action_size)
    def forward(self,state):
        return self.fc3(torch.rrelu(self.fc2(torch.rrelu(self.fc1(state)))))


