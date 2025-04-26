
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, n, in_features, out_features, self_connection=True, bias=True):
        '''n: 自旋个数,
           n*in: 总的输入个数,
           n*out: 总的输出个数,
         '''
        super(MaskedLinear, self).__init__(n * in_features, n * out_features, bias)
        #定义一个名为mask个的buffer     
        if self_connection:
            self.register_buffer('mask', torch.tril(torch.ones(n, n)))#注意 pytorch中是用行向量乘W.T定义的线性运算
        else:
            self.register_buffer('mask', torch.tril(torch.ones(n, n), diagonal=-1))
        self.mask = torch.cat([self.mask] * in_features, dim=1)
        self.mask = torch.cat([self.mask] * out_features, dim=0)
        self.weight.data *= self.mask
        if n !=1 :
            self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
    def forward(self, input):
            return F.linear(input, self.weight*self.mask, self.bias)
    

class GND(nn.Module):
    def __init__(self, m, k, device="cpu", dtype="float", activator='tanh'):
        super(GND, self).__init__()
        self.device, self.dtype = device, dtype
        self.m, self.k = m, k # m: number of stabilizers ; k: number of logical qubits

        if activator=='tanh':
            self.activator = nn.Tanh()
        elif activator=='relu':
            self.activator = nn.ReLU()
        elif activator=='sigmoid':
            self.activator = nn.Sigmoid()
        
    def Construct_CNN_block(self, L, depth=2, channels=3):
        net = []
        net.extend([
                nn.Conv2d(
                    in_channels=1,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                self.activator,
                # nn.MaxPool2d(kernel_size=2, stride=1),
            ])
        for i in range(depth-1):
            net.extend([
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                self.activator,
                # nn.MaxPool2d(kernel_size=2, stride=1),
            ])
        self.cnet = nn.Sequential(*net).to(self.device).to(self.dtype)
        self.linear = nn.Sequential(
            nn.Linear(channels*L**2, 64),
            self.activator,
            nn.Linear(64, 4**self.k)
            ).to(self.device).to(self.dtype)
       

    
    def Construct_MLP_block(self, depth, hiddens):
        net = []
        net.extend([nn.Linear(self.m, hiddens), self.activator])
        for i in range(depth-1):
            net.extend([nn.Linear(hiddens, hiddens), self.activator])
        # net.extend([nn.Linear(hiddens, self.k)])
        self.cnet =  nn.Sequential(*net).to(self.device).to(self.dtype)
        self.linear = nn.Linear(hiddens, 4**self.k).to(self.device).to(self.dtype)
    
    def Construct_GCN_block():
        None
    
    def Construct_MADE_block(self, n, width, depth):
        self.n=n
        net = []
        net.extend([
            MaskedLinear(n, 1, 1 if depth==0 and width==1 else width, False), 
            self.activator,
            ])
        for i in range(depth-1):
            net.extend([
                MaskedLinear(n, width, width, True, True),
                self.activator,
                ])
        if width != 1:
           net.extend([
                MaskedLinear(n, width, 1, True, True),
                self.activator,
                ])
        net.pop()
        net.extend([nn.Sigmoid(),])
        self.anet = nn.Sequential(*net).to(self.device).to(self.dtype)
        
    
    def Construct_TraDE_block():
        None

    def Classification_forward(self, x):
        x = self.cnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        # x = F.softmax(x, dim=1)
        return x

    def Autoregressive_forward(self, x):
        return self.anet(x)
    

    def Partially_Generate_forward(self, condition, device, dtype, cir_noise=False):
        if cir_noise == False:
            k = 2*self.k
        else:
            k = self.k
        dim = condition.dim()
        with torch.no_grad():
            if dim > 1 :
                m = condition.size(1)
                x = torch.zeros(condition.size(0), self.n, device=device, dtype=dtype)
            else:
                m = condition.size(0)
                x = torch.zeros(1, self.n, device=device, dtype=dtype)
            
            # print(x.size(), m)
            x[:, :m] = condition
            for i in range(k):
                s_hat = self.Autoregressive_forward(x)
                x[:, m+i] = torch.floor(2*s_hat[:, m+i])
        return x[:, m:]
    
    def log_prob(self, samples):
        a = 1e-30
        s = self.Autoregressive_forward(samples)
        log_p = (torch.log(s+a) * samples + torch.log(1 - s+a) * (1 - samples)).sum(dim=1)
        return log_p

    def num_para(self, net):
        n = sum([para.nelement() for para in net.parameters()])
        return n

if __name__ == '__main__':
    import argparse
    import sys
    from os.path import abspath, dirname
    sys.path.append(abspath(dirname(__file__)).strip('module'))
    from module import read_code, Loading_code, Errormodel, mod2

    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type=str, default='generate')
    parser.add_argument("-d", type=int, default=3)
    parser.add_argument("-k", type=int, default=1)
    parser.add_argument("-er", type=float, default=0.189)
    parser.add_argument("-trials", type=int, default=10000)
    parser.add_argument("-epoch", type=int, default=10000)
    parser.add_argument("-batch", type=int, default=10000)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-device", type=str, default='cuda:3')
    parser.add_argument("-depth", type=int, default=3)
    parser.add_argument("-hiddens", type=int, default=64) # para of mlp
    parser.add_argument("-width", type=int, default=5) # para of made
    args = parser.parse_args()

    mod2 = mod2()

    def reshape_syndrome(s, L):
        m = s.size(1)
        x = torch.zeros(s.size(0), 2*m+1, device=s.device, dtype=s.dtype)
        for i in range(m):
            x[:, 2*i+1] = s[:, i]
        x = x.view(s.size(0), 1, L, L)
        return x
    
    def oh_logical_label(l):
        ohl = torch.zeros(l.size(0), 4, device=l.device, dtype=l.dtype)
        index = l[:, 0]+2*l[:, 1]
        ohl.scatter_(1, index.view(-1, 1), 1)
        return ohl

    d = args.d
    k=args.k
    L = d+d-1
    info = read_code(d=d, k=k, seed=0, c_type='sur')
    Code = Loading_code(info)
    # print(Code.g_stabilizer)

    er = args.er
    trials = args.trials
    E = Errormodel(er, e_model='dep')
    errors = E.generate_error(Code.n,  m=trials, seed=0)
    
    # print(syndrome)
    # print(logical_label)
    
    device = args.device
    dtype=torch.float32
    syndrome0 = mod2.commute(errors, Code.g_stabilizer).to(device).to(dtype)
    logical_label = mod2.commute(errors, Code.logical_opt).to(device)
    x = reshape_syndrome(syndrome0, L).to(device).to(dtype)
    ohl = oh_logical_label(logical_label)
    coset = torch.argmax(ohl, dim=1).to(device)
    
    # print(x)


    epoch = args.epoch
    batch = args.batch
    lr  = args.lr
    

    def Classification_training(m = Code.m, k=k, L=L, epoch=epoch, batch=batch, lr=lr, device=device, dtype=dtype, depth=args.depth, hiddens=args.hiddens):
        net = GND(m, k, device=device, dtype=dtype, activator='tanh')
        net.Construct_MLP_block(depth=depth, hiddens=hiddens)
        n = net.num_para(net.cnet)+net.num_para(net.linear)
        print(net.cnet)
        # net.Construct_CNN_block(L=5, depth=2)
        optimizer = torch.optim.Adam(net.cnet.parameters(), lr=lr)
        his = []
        for l in range(epoch):
            ers = E.generate_error(Code.n, m=batch, seed=False)

            syndrome = mod2.commute(ers, Code.g_stabilizer).to(device).to(dtype)
            logical = mod2.commute(ers, Code.logical_opt).to(device).to(torch.long)
            label = oh_logical_label(logical).to(dtype)
            '''mlp out'''
            out = net.Classification_forward(syndrome)
            '''cnn out'''
            # input = reshape_syndrome(syndrome, L)
            # out = net.Classification_forward(input)

            loss = F.cross_entropy(out, label)#- torch.einsum('ab, ab -> a', (label, torch.log(out))).mean()#

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (l+1) % 100 == 0:
                
                print('number of parametters:', n)
                # print(F.softmax(net.Classification_forward(syndrome0)))
                '''mlp decoding'''
                pred_coset = torch.argmax(F.softmax(net.Classification_forward(syndrome0)), dim=1)
                '''cnn decoding'''
                # pred_coset = torch.argmax(F.softmax(net.Classification_forward(x)), dim=1)
                # print(pred_coset)
                # print(coset)
                fail_number = torch.count_nonzero(coset-pred_coset)
                logical_error_rate = fail_number/trials
                print(l+1, logical_error_rate)
                his.append(logical_error_rate.cpu().item())
        print(his)
    def Generative_training(m=Code.m, k=k, epoch=epoch, batch=batch, lr=lr, device=device, dtype=dtype, depth=args.depth, width=args.width):
        ni = m+2*k
        net = GND(m, k, device=device, dtype=dtype, activator='tanh')
        net.Construct_MADE_block(ni, width=width, depth=depth)
        n = net.num_para(net.anet)
        optimizer = torch.optim.Adam(net.anet.parameters(), lr=lr)
        # scheduler = StepLR(optimizer, step_size=2000, gamma=0.9)
        his = []
        for l in range(epoch):
            ers = E.generate_error(Code.n, m=batch, seed=False)
            configs = mod2.commute(ers, torch.vstack([Code.g_stabilizer, Code.logical_opt])).to(device).to(dtype)
            logp = net.log_prob(configs)
            loss = torch.mean((-logp), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
            #    scheduler.step()
        
            if (l+1) % 100 == 0:
                '''correction'''
                
                print('number of parametters:', n)
                logical_pred = net.Partially_Generate_forward(syndrome0, device, dtype)
                num_fail = torch.count_nonzero(abs((logical_label - logical_pred)).sum(1))
                logical_error_rate = num_fail/trials
                print(l+1, logical_error_rate)
                his.append(logical_error_rate.cpu().item())
        print(his)
    
    if args.task == 'classify':
        Classification_training()
    elif args.task == 'generate':
        Generative_training()
    

