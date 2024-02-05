import torch
torch.save(model, './model.pth')
torch.save(model.state_dict(), './model_state.pth')

load_model = torch.load('model.pth')
model.load_state_dic(torch.load('model_state.pth'))