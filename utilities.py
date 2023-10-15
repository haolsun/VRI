import torch
import torch.nn.functional as F

def kl_loss(mu1, log_var1, mu2, log_var2):
    s = (log_var2 - log_var1) + (torch.pow(torch.exp(log_var1), 2) + torch.pow(mu1 - mu2, 2)) / (
                2 * torch.pow(torch.exp(log_var2), 2)) - 0.5
    return torch.mean(s)

def loss_function(logits, onehot_label):
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)
    loss = - torch.sum(log_prob * onehot_label) / logits.size(0)
    return loss

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy