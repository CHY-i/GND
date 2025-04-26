
from args import args
import torch
import time
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Errormodel, mod2, Loading_code, read_code, btype

def forward(n_s, m, van, syndrome, device, dtype, k=1, n_type='made'):
    if n_type =='made':
        condition = syndrome*2-1
        x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    elif n_type == 'trade':
        condition = syndrome
        x = van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k)
    x = x[:, m:m+2*k]
    return x

if __name__ == '__main__':
    from module import Surfacecode
    n_type = 'made'
    save = args.save
    
    trials = args.trials
    n = args.n
    d, k, device, seed = args.d, args.k, args.device, args.seed
    epoch, batch, lr = args.epoch, args.batch, args.lr

    er = args.er
    c_type = args.c_type
    e_model = args.e_model

    if c_type == 'qcc':
        path = abspath(dirname(__file__))+'/net/code_capacity/made_'+c_type+'_n{}_k{}_er{}.pt'.format(n, k, er)
    else:
        path = abspath(dirname(__file__))+'/net/code_capacity/'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}_mid1.pt'.format(d, k, seed, er)
    
    net = torch.load(path)
    dtype=net.deep_net[0].weight.dtype

    mod2 = mod2(device=device, dtype=dtype)

    info = read_code(d=d, k=k, seed=seed, c_type=c_type, n=n)
    Code = Loading_code(info)
    
    e1 = Code.pure_es 
    for i in range(Code.m):
        conf = mod2.commute(e1[i], e1)
        idx = conf.nonzero().squeeze()
        sta = Code.g_stabilizer[idx]
        e1[i] = mod2.opts_prod(torch.vstack([e1[i], sta]))

    g = torch.vstack([e1, Code.logical_opt, Code.g_stabilizer])##
    
    
    
    E = Errormodel(er, e_model=e_model)
    errors = E.generate_error(Code.n,  m=trials, seed=seed)
    
    syndrome = mod2.commute(errors, Code.g_stabilizer)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)
    
    ni = Code.m+2*k
    
    van = MADE(n=ni, depth=args.depth, width=args.width).to(device).to(dtype)
    van.deep_net = net.deep_net.to(device).to(dtype)

    lconf = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type)
    '''correction'''
    l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
    recover = mod2.opt_prod(pe, l)
    check = mod2.opt_prod(recover, errors)
    commute = mod2.commute(check, Code.logical_opt)
    fail = torch.count_nonzero(commute.sum(1))
    logical_error_rate = fail/trials
    print(logical_error_rate)

    optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
    # loss_his = []
    # lo_his = []
    for j in range(epoch):

        ers = E.generate_error(Code.n, m=batch, seed=False)

        configs = E.configs(sta=Code.g_stabilizer, log=Code.logical_opt, pe=e1, opts=ers).to(device)

        logp = van.log_prob((configs[:, :ni])*2-1)
       
        loss = torch.mean((-logp), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if optimizer.state_dict()['param_groups'][0]['lr'] > 0.00004 :
           scheduler.step()
        if (j+1)%100==0 and e_model != 'ising':
        #     logq = E.log_probability(opts=ers, device=device, dtype=dtype)
        #     KL = torch.mean(logq-logp, dim=0)
            print(loss)
            # loss_his.append(KL)
        if (j+1) % 1000 == 0:
            print(j)
            lconf = forward(n_s=trials, m=Code.m, van=van, syndrome=syndrome, device=device,dtype=dtype, k=k, n_type=n_type)
            #print(lconf)

            '''correction'''
            l = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
            recover = mod2.opt_prod(pe, l)
            check = mod2.opt_prod(recover, errors)
            commute = mod2.commute(check, Code.logical_opt)
            fail = torch.count_nonzero(commute.sum(1))
            logical_error_rate = fail/trials
            print(logical_error_rate)
            
            # lo_his.append(logical_error_rate)
    # print(loss_his)
    # print(lo_his)
    # torch.save((loss_his, lo_his), abspath(dirname(__file__))+'/his.pt')
    if save == True:
        if c_type == 'qcc':
            path = abspath(dirname(__file__))+'/net/code_capacity/made_'+c_type+'_n{}_k{}_er{}C.pt'.format(n, k, er)
        else:
            path = abspath(dirname(__file__))+'/net/code_capacity/made_'+c_type+'_d{}_k{}_seed{}_er{}C.pt'.format(d, k, seed, er)
        # if exists(path):
        #     None
        # else:          
        torch.save(van, path)
