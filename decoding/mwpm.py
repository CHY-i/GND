from args import args
from pymatching import Matching
import numpy as np
import torch
import time
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import Loading_code, read_code, Hx_Hz, Errormodel, mod2, Abstractcode

d, k, seed, c_type = args.d, args.k, args.seed, args.c_type
trials = args.trials
device, dtype = 'cpu', torch.float32
e_model = args.e_model
error_seed = args.error_seed
mod2 = mod2(device=device, dtype=dtype)
if c_type == 'drsur':
    defect_g = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]#[14, 16, 18]
    info = read_code(d=7, k=1, seed=seed, c_type='rsur')
    oCode = Loading_code(info)
    g = oCode.g_stabilizer[defect_g,:]
    code = Abstractcode(g_stabilizer=g)
else:    
    info = read_code(d=d, k=k, seed=seed, c_type=c_type)
    code = Loading_code(info)
n = code.n
PCM = code.PCM.cpu().numpy()
#print(PCM)
hx, hz = Hx_Hz(code.g_stabilizer)
hx, hz = hx.cpu().numpy(), hx.cpu().numpy()
#print(hx)
#print(hz)

l1 = mod2.rep(code.logical_opt).int().numpy()
l = np.zeros_like(l1)
l[:, :n], l[:, n:] = l1[:, n:], l1[:, :n]
#print(l)



    
L = []
error_rate = torch.logspace(-1.5, -0.5, 10)#
print(error_rate)
tt = torch.zeros(len(error_rate))
for i in range(len(error_rate)):
    '''generate error'''
    E = Errormodel(error_rate[i], e_model=e_model)
    errors = E.generate_error(code.n,  m=trials, seed=error_seed)
    if e_model == 'dep':
        er = 2*error_rate[i]/3
    elif e_model == 'dep2':
        er = 8*error_rate[i]/15

    weights = torch.ones(2*code.n)*torch.log((1-er)/er)
    Decoder = Matching(PCM, weights=weights)
    syndrome = mod2.commute(errors, code.g_stabilizer)
    pe = E.pure(code.pure_es, syndrome,device=device, dtype=dtype)
    error = mod2.rep(errors).squeeze().int().numpy()
    syndrome = syndrome.numpy()

    correct_number = 0
    t = 0
    for j in range(trials):
        e = error[j]
        s = syndrome[j]

        t1 = time.time()
        #print(s)
        recover = Decoder.decode(s)
        check = (e + recover)%2
        s = np.sum((check @ l.T) %2)
        t2 = time.time()
        t = t+(t2-t1)
        if s == 0:
            correct_number+=1
        
    lorate = 1 - correct_number/trials
    ta = t#/trials
    print(lorate)
    print(ta)
    tt[i] = ta
    L.append(int(10000*lorate)/10000)
print(L)
print(tt.mean().item(), tt.std().item())    




