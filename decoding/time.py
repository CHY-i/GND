from args import args
import torch
import time
import sys
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Errormodel, mod2, Loading_code, read_code
torch.backends.cudnn.enable =True

def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
    if n_type =='made':
        condition = syndrome*2-1
        x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    elif n_type == 'trade':
        condition = syndrome
        x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
    x = x[:, m:m+2*k]
    return x

'''Hyper Paras:'''
if args.dtype == 'float32':
    dtype = torch.float32
elif args.dtype == 'float64':
    dtype = torch.float64
device = args.device
trials = args.trials
'''Code Paras:'''
c_type, n, d, k, seed = args.c_type, args.n, args.d, args.k, args.seed
print(n)
'''Net Paras:'''
n_type, e_model = 'made', args.e_model
er = args.er
times = 100


if c_type == 'qcc':
    path = abspath(dirname(__file__))+'/net/code_capacity/'+n_type+'_'+c_type+'_n{}_k{}_er{}.pt'.format(n, k, er)

else:
    path = abspath(dirname(__file__))+'/net/code_capacity/'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}_{}.pt'.format(d, k, seed, er, e_model)
print(path)
error_rate=0.189



'''Loading Code'''
info = read_code(d, k, seed, c_type=c_type, n=n)
Code = Loading_code(info)

'''Loading Net'''
net = torch.load(path, map_location='cpu')
if n_type == 'made':
    van = MADE(Code.n+k, 0, 1).to(device).to(dtype)
    van.deep_net = net.to(device).to(dtype)

mod2 = mod2(device=device, dtype=dtype)





lo_rate = []

if e_model == 'dep':
    E = Errormodel(error_rate, e_model='dep')#error_rate[i]
    errors = E.generate_error(Code.n, m=trials, seed=0).squeeze()
    syndrome = mod2.commute(errors, Code.g_stabilizer)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)

for j in range(100):
    _ = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type)

correct_number = 0

tt = torch.zeros(times)
for j in range(times):
    '''forward to get configs'''
    t1 = time.time()
    lconf = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type)
    # print(lconf.size())
    t3 = time.time()
    '''correction'''
    l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
    recover = mod2.opt_prod(pe, l)
    check = mod2.opt_prod(recover, errors)
    commute = mod2.commute(check, Code.logical_opt)
    t4 = time.time()
    if trials ==1 :
        fail = commute.sum()
        #print(fail)
        if fail == 0 :
            correct_number += 1
        logical_error_rate = 1-correct_number/times

    else:
        fail = torch.count_nonzero(commute.sum(1))
        logical_error_rate = (fail/trials).item()

    t2 = time.time()
    print(t3-t1, t4-t3)
    tt[j] = (t3-t1)
print(logical_error_rate)
print(tt[int(times/2):].mean().item(), tt[int(times/2):].std().item())
