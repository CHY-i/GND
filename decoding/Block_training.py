
import torch.nn.functional
from args import args
import torch
import time
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import TraDE, Errormodel, mod2, Loading_code, read_code

def block_spin(configs, ns, nl, nb, block_l=True):
    batch = configs.size(0)
    s = configs[:, :ns]
    l = configs[:, ns:]
    a = ns%nb
    if a !=0:
        s = torch.hstack([s, torch.zeros(batch, nb-a, device=s.device, dtype=s.dtype)])
    s = s.reshape(batch, -1, nb)
    v = torch.tensor([2**i for i in range(nb)], device=s.device, dtype=s.dtype)
    s = torch.einsum('abc, c -> ab', (s, v))

    if block_l:
        b = nl%nb
        if b !=0:
            l = torch.hstack([l, torch.zeros(batch, nb-b, device=l.device, dtype=l.dtype)])
        l = l.reshape(batch, -1, nb)
        l = torch.einsum('abc, c -> ab', (l, v))
    # print(ns, nl)
    # print(s)
    # print(l)
    block_configs = torch.hstack([s, l])
    return block_configs



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

    nb = 4



    mod2 = mod2(device=device, dtype=dtype)

    info = read_code(d=d, k=k, seed=seed, c_type=c_type, n=n)
    Code = Loading_code(info)
    
    e1 = Code.pure_es 
    g = torch.vstack([e1, Code.logical_opt, Code.g_stabilizer])##
    


    er = args.er
    E = Errormodel(er, e_model=e_model)
    errors = E.generate_error(Code.n,  m=3, seed=seed)

    syndrome = mod2.commute(errors, Code.g_stabilizer)
    lconfigs = mod2.commute(errors, Code.logical_opt)
    pe = E.pure(Code.pure_es, syndrome, device=device, dtype=dtype)
    
    ni = Code.m+2*k
    ns, nl = Code.m, 2*k

    nbs, nbl = int(ns/nb) if ns%nb ==0 else int(ns/nb)+1, int(nl/nb) if nl%nb ==0 else int(nl/nb)+1
    # print(nbs, nbl)

    configs = torch.hstack([syndrome, lconfigs])
    bc = block_spin(configs=configs, ns=ns, nl=nl, nb=nb)
    # print(bc[:, ])

    # mask = torch.zeros(3, bc.size(1), 2**nb)
    # print(mask.size())
    # for i in range(bc.size(1)):
    #     mask[:, bc[:, i].long()] = 1.0
    # print(mask.size())
    # print(mask.nonzero())


    kwargs_dict = {
        'n': nbs+nbl,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'n_layers': args.n_layers,
        'device': args.device,
        'dropout': 0, 
        'nb': nb
        }
    van = TraDE(**kwargs_dict).to(device).to(dtype)

    # batch = 3

    optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
    # loss_his = []
    # lo_his = []
    for l in range(epoch):
        ers = E.generate_error(Code.n, m=batch, seed=False)

        configs = mod2.commute(ers, torch.vstack([Code.g_stabilizer, Code.logical_opt])).to(device).to(dtype)
        block_configs = block_spin(configs, ns, nl, nb)

        logp = van.log_prob(block_configs)
       
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
            print(l)
            l_conf = block_configs[:, nbs:]
            l_predict = van.partial_forward(batch, nbl, nl%nb, block_configs[:, :nbs], device, dtype)
            # print(lconf)
            logical_error_rate =  (l_predict-l_conf).sum(1).count_nonzero()/batch
            print(logical_error_rate)
    #         '''correction'''
    #         L = mod2.confs_to_opt(confs=lconf, gs=Code.logical_opt)
    #         recover = mod2.opt_prod(pe, L)
    #         check = mod2.opt_prod(recover, errors)
    #         commute = mod2.commute(check, Code.logical_opt)
    #         fail = torch.count_nonzero(commute.sum(1))
    #         logical_error_rate = fail/trials
    #         print(l, logical_error_rate)
    #         # lo_his.append(logical_error_rate)
    # # print(loss_his)
    # # print(lo_his)
    # # torch.save((loss_his, lo_his), abspath(dirname(__file__))+'/his.pt')
    if save == True:
        if c_type == 'qcc':
            path = abspath(dirname(__file__))+'/net/block/'+n_type+'_'+c_type+'_n{}_k{}_er{}.pt'.format(n, k, er)
        else:
            path = abspath(dirname(__file__))+'/net/block/'+n_type+'_'+c_type+'_d{}_k{}_seed{}_er{}_{}.pt'.format(d, k, seed, er, e_model)

        if exists(path):
            None
        else:          
            torch.save(van, path)
       