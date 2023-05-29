import torch

class Agent:
    def __init__(self) -> None:
        pass
    def select_action(self, state):
        pass
        
class VictimAgent(Agent):
    def __init__(self, action_selection_model, device) -> None:
        super().__init__()
        self.device = device
        # the model object and parameter should be loaded before the victim agent is initiated
        self.select_action_model = action_selection_model

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).view(25).unsqueeze(0)
        return self.select_action_model(state).max(1)[1].view(1, 1).item()

    
class AttackerAgent(Agent):
    def __init__(self, id) -> None:
        super().__init__()
        self.id = id
        # the model object and parameter should be loaded before the victim agent is initiated
        # self.select_action_model = action_selection_model

    def select_action(self, a):
        pass
        # return self.select_action_model(state)