import torch
class environment():
    def __init__(self,agent):
        self.agent=agent
        self.position=None
        self.actions_space=[torch.tensor([1.,0.]),torch.tensor([0.,1.]),torch.tensor([-1.,0.]),torch.tensor([0.,-1.])]
    def get_reward(self):
        #返回奖励值与游戏结束与否
        if(self.position[0][0]> 0 and self.position[0][0]<8 and self.position[0][1]>0 and self.position[0][1]<4):
            if(torch.any(self.position!=torch.tensor([[2.,1.]])) and
               torch.any(self.position!=torch.tensor([[3.,1.]])) and
               torch.any(self.position!=torch.tensor([[4.,1.]])) and
               torch.any(self.position!=torch.tensor([[5.,1.]])) and
               torch.any(self.position!=torch.tensor([[6.,1.]]))):
                if(torch.all(self.position==torch.tensor([[7.,1.]]))):
                    #print("a")
                    return torch.tensor(100.),True
                else:
                    #print("b")
                    return torch.tensor(1.),False
            else:
                #print("c")
                return torch.tensor(-30.),True
        else:
            #print("d")
            return torch.tensor(-30.),True
    def execute_step(self,action):
        self.position=self.position+self.actions_space[action.item()]
        return self.get_reward()
    
    def run(self):
        self.position=torch.tensor([1.,1.])
        done=False
        while True:
            if done:
                break
            position=self.position
            index_tensor=self.agent.choose_action(self.position)
            reward,done=self.execute_step(index_tensor)
            next_position=self.position
            value=self.agent.network.forward(position.unsqueeze(0))[0,index_tensor.item()]
            next_value=torch.max(self.agent.network.forward(next_position.unsqueeze(0)).view(-1))
            diff_value=torch.abs(reward+self.agent.gamma*next_value-value)
            if self.agent.memory.memory_size()==0:
                self.agent.memory.add_sample((position,next_position,index_tensor,reward,done),diff_value)
            else:
                if (position,next_position,index_tensor,reward,done) not in self.agent.memory.replay_buffer:
                    self.agent.memory.add_sample((position,next_position,index_tensor,reward,done),diff_value)
                else:
                    False
    def verify(self):
        self.position=torch.tensor([1.,1.])
        done=False
        total_reward=0
        while True:
            if done:
                break
            index_tensor=self.agent.choose_action(self.position)
            reward,done=self.execute_step(index_tensor)
            total_reward+=reward
        return total_reward
        
        