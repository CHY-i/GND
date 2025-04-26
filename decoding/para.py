from args import args
import torch
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))


c_type, n, d, k, seed = 'qcc', 60, 4, 8, 0
'''Net Paras:'''
n_type, e_model = 'made', args.e_model
er = 0.1#args.er
'''seed for sampling errors:'''
error_seed = args.error_seed
if c_type == 'qcc':
    path = abspath(dirname(__file__))+'/net/code_capacity/'+n_type+'_'+c_type+'_n{}_k{}_er{}.pt'.format(n, k, er)
else:
    path = abspath(dirname(__file__))+'/net/code_capacity/'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}'.format(d, k, seed, er).format(d, k, seed, er)+'_'+e_model+'.pt'

path = abspath(dirname(__file__))+'/net/cir/d5_r5c1.pt'
net = torch.load(path)
print(net)
print(net.n)
print(net.depth)
print(net.width)
n = sum([para.nelement() for para in net.deep_net.parameters()])
print(n)

# van = MADE(n=50, depth=3, width=30, residual=False)
# n = sum([para.nelement() for para in van.deep_net.parameters()])
# print(n)