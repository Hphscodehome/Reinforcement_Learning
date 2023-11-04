import torch
class agent():
    def __init__(self,memory,network,action_size,state_size):
        self.memory=memory
        self.network=network
        self.action_size=action_size
        self.batch_size=50
        self.state_size=state_size
        self.gamma=0.9
    
    def choose_action(self,state,epsilon=0.1):
        if(torch.rand((1,))>epsilon):
            state=torch.FloatTensor(state)
            state=state.unsqueeze(0)
            values=self.network(state)
            action=torch.argmax(values.view(-1))
        else:
            action=torch.randint(0,self.action_size,(1,))
        #返回动作的索引，张量
        return action
    def train_batch(self,inputs,outputs):
        loss=torch.nn.MSELoss()(self.network.forward(inputs),outputs)
        optimizer=torch.optim.Adam(self.network.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def experience_replay(self):
        print("Trainging")
        samples=self.memory.sample(self.batch_size)
        if(samples!=None):
            last_states, next_states, actions, rewards,flags = samples
            with torch.no_grad():
                inputs = torch.stack(last_states,dim=0)#last_states必须要是张量
                n_states = torch.stack(next_states,dim=0)
                q_values = self.network.forward(inputs)
                next_q_values = self.network.forward(n_states)
                for i in range(self.batch_size):
                    if(flags[i]==True):
                        q_values[i,0,actions[i].item()]=rewards[i]
                    else:
                        q_values[i,0,actions[i].item()]=rewards[i]+self.gamma*torch.max(next_q_values[i,:,:])
            self.train_batch(inputs,q_values.detach().clone())
        else:
            False
    
    
