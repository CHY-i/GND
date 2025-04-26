
from args import args
import torch
import time
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, TraDE_binary, NADE,  Errormodel, mod2, Loading_code, read_code, Abstractcode

def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
    if n_type =='made':
        condition = syndrome*2-1
        x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    elif n_type == 'trade' or 'nade':
        condition = syndrome
        x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
    x = x[:, m:m+2*k]

    return x

if __name__ == '__main__':
    from module import Surfacecode
    n_type = args.n_type
    save = args.save
    if args.dtype == 'float32':
        dtype= torch.float32
    elif args.dtype == 'float64':
        dtype= torch.float64

    trials = args.trials
    n, d, k, device, seed =args.n, args.d, args.k, args.device, args.seed
    epoch, batch, lr = args.epoch, args.batch, args.lr
    
    c_type = args.c_type
    e_model = args.e_model

    

    mod2 = mod2(device=device, dtype=dtype)

    if c_type == 'drsur':
        defect_g = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]#[14, 16, 18]
        info = read_code(d=7, k=1, seed=seed, c_type='rsur', n=n)
        oCode = Loading_code(info)
        g = oCode.g_stabilizer[defect_g,:]
        Code = Abstractcode(g_stabilizer=g)
    else:    
        info = read_code(d=d, k=k, n=n, seed=seed, c_type=c_type)
        Code = Loading_code(info)
    
    e1 = Code.pure_es 
    for i in range(Code.m):
        conf = mod2.commute(e1[i], e1)
        idx = conf.nonzero().squeeze().cpu()
        sta = Code.g_stabilizer[idx]
        e1[i] = mod2.opts_prod(torch.vstack([e1[i], sta]))

    g = torch.vstack([e1, Code.logical_opt, Code.g_stabilizer])##
    


    er = args.er
    E = Errormodel(er, e_model=e_model)
    errors = E.generate_error(Code.n,  m=trials, seed=seed)

    syndrome = mod2.commute(errors, Code.g_stabilizer)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)
    
    ni = Code.m+2*k

    
    if n_type == 'made':
        van = MADE(n=ni, depth=args.depth, width=args.width, residual=False).to(device).to(dtype)
    elif n_type == 'nade':
        van = NADE(n=ni, hidden_dim=5000, device=device, dtype=dtype)
    elif n_type == 'trade':
        kwargs_dict = {
        'n': ni,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'n_layers': args.n_layers,
        'device': args.device,
        'dropout': 0, 
        }
        van = TraDE_binary(**kwargs_dict).to(device).to(dtype)



    optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.9)
    # loss_his = []
    # lo_his = []
    for l in range(epoch):
        ers = E.generate_error(Code.n, m=batch, seed=False)

        configs = E.configs(sta=Code.g_stabilizer, log=Code.logical_opt, pe=e1, opts=ers).to(device).to(dtype)

        if n_type == 'made':
            logp = van.log_prob((configs[:, :ni])*2-1)
        elif n_type == 'nade':
            logp = van.forward((configs[:, :ni]))
        elif n_type == 'trade':
            logp = van.log_prob((configs[:, :ni]))
       
        loss = torch.mean((-logp), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
           scheduler.step()
        if (l+1)%100==0 and e_model != 'ising':
        #     logq = E.log_probability(opts=ers, device=device, dtype=dtype)
        #     KL = torch.mean(logq-logp, dim=0)
            print(loss)
            # loss_his.append(KL)
        if (l+1) % 1000 == 0:
            lconf = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device, dtype=dtype, k=k, n_type=n_type)
            #print(lconf)

            '''correction'''
            L = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
            recover = mod2.opt_prod(pe, L)
            check = mod2.opt_prod(recover, errors)
            commute = mod2.commute(check, Code.logical_opt)
            fail = torch.count_nonzero(commute.sum(1))
            logical_error_rate = fail/trials
            print(l, logical_error_rate)
            # lo_his.append(logical_error_rate)
    # print(loss_his)
    # print(lo_his)
    # torch.save((loss_his, lo_his), abspath(dirname(__file__))+'/his.pt')
    if save == True:
        path = abspath(dirname(__file__))+'/net/code_capacity/'+n_type+'_'+c_type+'_n{}_d{}_k{}_seed{}_er{}_{}.pt'.format(n, d, k, seed, er, e_model)

        if exists(path):
            None
        else:          
            torch.save(van, path)
       