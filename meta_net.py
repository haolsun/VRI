import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class vri(MetaModule):
    def __init__(self, input, hidden1, hidden2, output):
        # 64, 128, 10
        super(vri, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden2)
        self.linear3 = MetaLinear(hidden2, hidden2)
        self.linear_mean = MetaLinear(hidden2, output)
        self.linear_var = MetaLinear(hidden2, output)

        self.cls_emb = nn.Embedding(output, 64)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear3.weight)
        self.linear3.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_var.weight)
        self.linear_var.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        h3 = self.tanh(self.linear3(h2))
        mean = self.linear_mean(h3)
        std = self.linear_var(h3)
        return mean, std

    def forward(self, dict):
        if 'target' in dict:
            target = self.cls_emb(dict['target'])
            x = torch.cat([dict['feature'], target], dim=-1)
        else:
            x = torch.cat([dict['feature'], torch.zeros(dict['feature'].shape[0], 64).cuda()], dim=-1)

        mean, log_var = self.encode(x)  # [100, 10]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # s = 1 + 2 * torch.log(std) - mean.pow(2) - std.pow(2)
        # kl_loss = -0.5 * torch.mean(s)

        # return mean, std, F.sigmoid(mean + std*eps), torch.norm(std,2,dim=1).mean()
        return mean, std, F.sigmoid(mean + std * eps) #, torch.max(std, dim=0).values.mean()



class vri_prior(MetaModule):
    def __init__(self, input, hidden1, output):
        # 64, 128, 10
        super(vri_prior, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden1)
        self.linear_mean = MetaLinear(hidden1, output)
        self.linear_var = MetaLinear(hidden1, output)

        self.cls_emb = nn.Embedding(output, 64)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_var.weight)
        self.linear_var.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.tanh(self.linear_mean(h2))
        log_var = self.tanh(self.linear_var(h2))
        return mean, log_var

    def forward(self, feat):

        mean, log_var = self.encode(feat) # [100, 10]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        s = 1 + log_var - mean.pow(2) - log_var.exp()
        kl_loss = -0.5 * torch.mean(s)

        return mean, log_var, F.sigmoid(mean + std*eps) #, kl_loss
        #return mean, log_var, F.sigmoid(mean + std*eps)*2-1 #, kl_loss


class vri_dec(MetaModule):
    def __init__(self, input, hidden1, output):
        # 64, 128, 10
        super(vri_dec, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden1)
        self.linear_mean = MetaLinear(hidden1, output)

        self.cls_emb = nn.Embedding(output, 64)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.linear_mean(h2)
        return mean

    def forward(self, feat, target):
        target = self.cls_emb(target)

        x = torch.cat([feat, target], dim=-1)
        mean = self.encode(x) # [100, 10]
        return F.sigmoid(mean), None








