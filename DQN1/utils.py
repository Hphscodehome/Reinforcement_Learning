import torch
import torch.nn as nn
class Memory():
    #memory流程为：
    #采集阶段：添加sample
    #学习阶段：读取sample调整网络，读取buffer调整权重，更新权重赋值
    #初始化
    def __init__(self,size):
        self.size=size
        self.replay_buffer=[]#样本
        self.buffer_bias=[]#样本偏差
    #添加样本
    def add_sample(self,sample,bias):
        if len(self.replay_buffer)>self.size:
            self.replay_buffer.pop(0)
            self.buffer_bias.pop(0)
        self.replay_buffer.append(sample)
        self.buffer_bias.append(bias)
        
    #优先经验回放
    def sample(self,batch_size):
        if(len(self.replay_buffer)<batch_size):
            return None
        else:
            prob=torch.softmax(torch.log(torch.tensor(self.buffer_bias)),dim=0)
            print("prob:",prob)
            batch_size=batch_size
            index=torch.multinomial(prob,batch_size,replacement=True)
            print(index)
            weight=[]
            out=[]
            for i in index:
                out.append(self.replay_buffer[i])
                weight.append(1/torch.sqrt(len(self.replay_buffer)*prob[i]))
            samples=list(zip(*out))#返回状态动作列表，还需要返回权重
            return samples,weight
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

if __name__=='__main__':
    state_size=4
    action_size=2
    memory=Memory(100)
    q_network=Q_network(state_size,action_size)
    optimizer=torch.optim.Adam(q_network.parameters(),lr=0.001)
    for i in range(100):
        state=torch.rand(1,state_size)
        action=torch.randint(0,action_size,(1,))
        next_state=torch.rand(1,state_size)
        reward=torch.rand(1,)
        done=torch.randint(0,2,(1,))
        with torch.no_grad():
            bias=torch.abs(q_network.forward(state)[0][action.item()]-reward-0.9*torch.max(q_network.forward(next_state).view(-1)))
        memory.add_sample((state,action,next_state,reward,done),bias)
    for k in range(20):
        (list_state,list_action,list_next_state,list_reward,list_done),list_weights = memory.sample(10)
        states=torch.cat(list_state)
        print(states,"states type",states.shape)
        next_states=torch.cat(list_next_state)
        actions=torch.cat(list_action)
        rewards=torch.cat(list_reward)
        dones=torch.cat(list_done)
        weights=torch.tensor(list_weights)
        with torch.no_grad():
            q_values=q_network(states)
            next_q_values=q_network(next_states)
            print("torch.max(next_q_values,dim=1)[0]",torch.max(next_q_values,dim=1),torch.max(next_q_values,dim=1)[1])
            target_q_values=q_values.detach().clone()
            target_q_values[range(len(target_q_values)),actions]=rewards+0.9*(1-dones)*torch.max(next_q_values,dim=1)[0]
            print("target_q_values,q_values",target_q_values,q_values)
        loss=torch.sum(weights.reshape(-1,1)*(target_q_values-q_network(states))**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for paras in q_network.parameters():
            print(paras)

        

    


