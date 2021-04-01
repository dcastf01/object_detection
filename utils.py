import torch


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            # scores = torch.sigmoid()
            predictions = torch.log_softmax(model(x),dim=1)
            _, predictions = torch.max(predictions, dim = 1) 
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.shape[0]

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f} %')

    model.train()
    
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
