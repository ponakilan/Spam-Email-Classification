import torch
from sklearn.metrics import accuracy_score


def accuracy(dataloader, model, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            classes = []
            for output in outputs:
                if output[0] >= 0.5:
                    classes.append(1)
                else:
                    classes.append(0)
            scores.append(accuracy_score(Y.cpu().numpy().flatten().tolist(), classes))
    return sum(scores) / len(scores)
