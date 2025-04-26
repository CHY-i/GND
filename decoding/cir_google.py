from args import args
import torch
import time
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Data, mod2

basis, d, r = 'X', 5, 5
center = '5_5'

path_s = abspath(dirname(__file__)).strip('decoding') + 'google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/detection_events.b8'
path_l = abspath(dirname(__file__)).strip('decoding') + 'google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/obs_flips_actual.01'
path_dem = abspath(dirname(__file__)).strip('decoding') + 'google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/pij_from_even_for_odd.dem'
data = Data(d, r, path_s, path_l)


syndromes = torch.tensor(data.syndromes())*1.
logicals = torch.tensor(data.logical_flip())*1.
logicals_pre_TN = torch.tensor(data.logical_flip(path = abspath(dirname(__file__)).strip('decoding') + '/google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/obs_flips_predicted_by_tensor_network_contraction.01'))*1.
logicals_pre_MWPM =  torch.tensor(data.logical_flip(path = abspath(dirname(__file__)).strip('decoding') + '/google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/obs_flips_predicted_by_pymatching.01'))*1.
logicals_pre_BM =  torch.tensor(data.logical_flip(path = abspath(dirname(__file__)).strip('decoding') + '/google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/obs_flips_predicted_by_belief_matching.01'))*1.
logicals_pre_CM =  torch.tensor(data.logical_flip(path = abspath(dirname(__file__)).strip('decoding') + '/google_data/surface_code_b'+basis+'_d{}_r0{}_center_'.format(d, r)+center+'/obs_flips_predicted_by_correlated_matching.01'))*1.



n_data = syndromes.size(0)
input = torch.hstack((syndromes, logicals))
print(syndromes.size(), logicals.size())



def forward(n_s, m, van, syndrome, device, dtype, k=1):
    condition = syndrome*2-1
    x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    x = x[:, m:m+int(2*k)]
    return x

if __name__ == '__main__':
    from module import Surfacecode
    with open(path_dem, 'r') as file:
        dem = file.read()
    # print(dem)
    import stim
    dem = stim.DetectorErrorModel.from_file(path_dem)
    sampler = dem.compile_sampler()
    # samples = sampler.sample(2)
    # s = torch.hstack((torch.tensor(samples[0])*1.0, torch.tensor(samples[1])*1.0))
    # print(s.shape)

    ni = input.size(1)
    epoch = 500000
    batch = 10000
    training_size = 0
    test_size = input.size(0)-training_size
    print(training_size, test_size)
    lr = 0.001

    ltn = abs(logicals-logicals_pre_TN)[training_size:].sum()/test_size
    lmw = abs(logicals-logicals_pre_MWPM)[training_size:].sum()/test_size
    lbm = abs(logicals-logicals_pre_BM)[training_size:].sum()/test_size
    lcm = abs(logicals-logicals_pre_CM)[training_size:].sum()/test_size

    print(ltn, lmw, lbm, lcm)

    device='cuda:0'
    dtype = torch.float32
    
    input = input.to(device).to(dtype)
    logicals = logicals.to(device).to(dtype)

    
    import time
    Ctraining = False
    decoding = True
    if Ctraining == False and decoding == False:
        van = MADE(n=ni, depth=3, width=40, residual=False).to(device).to(dtype)

        optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
        # loss_his = []
        # lo_his = []
        for l in range(epoch):
            
            
            samples = sampler.sample(batch)
            s = torch.hstack((torch.tensor(samples[0])*1.0, torch.tensor(samples[1])*1.0)).to(device).to(dtype)
            logp = van.log_prob((s*2-1))
        
            loss = torch.mean((-logp), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
                scheduler.step()
            
            if (l) % 1000 == 0:
                t0 = time.time()
                lconf = forward(n_s=50000, m=ni-1, van=van, syndrome=input[:, :-1], device=device, dtype=dtype, k=1/2)
                logical_error_rate = abs(logicals-lconf).sum()/50000
                t1 = time.time()
                print(t1-t0)
                print(logical_error_rate)
                if logical_error_rate < 0.1130:
                    break
        torch.save(van, abspath(dirname(__file__))+'/net/cir/d{}_r{}a.pt'.format(d, r))
    elif Ctraining == True and decoding == False:
        lr = 0.00009
        batch = 50000

        path = abspath(dirname(__file__))+'/net/cir/d{}_r{}c.pt'.format(d, r)
        net = torch.load(path)
        dtype=net.deep_net[0].weight.dtype

        van = MADE(n=ni, depth=args.depth, width=args.width).to(device).to(dtype)
        van.deep_net = net.deep_net.to(device).to(dtype)
        optimizer = torch.optim.Adam(van.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

        lconf = forward(n_s=50000, m=ni-1, van=van, syndrome=input[:, :-1], device=device, dtype=dtype, k=1/2)
        logical_error_rate = abs(logicals-lconf).sum()/50000
        print(logical_error_rate)

        for l in range(epoch):
            
            samples = sampler.sample(batch)
            s = torch.hstack((torch.tensor(samples[0])*1.0, torch.tensor(samples[1])*1.0)).to(device).to(dtype)
            logp = van.log_prob((s*2-1))
        
            loss = torch.mean((-logp), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0001 :
                scheduler.step()

            
            if (l) % 1000 == 0:
                t0 = time.time()
                lconf = forward(n_s=50000, m=ni-1, van=van, syndrome=input[:, :-1], device=device, dtype=dtype, k=1/2)
                logical_error_rate = abs(logicals-lconf).sum()/50000
                t1 = time.time()
                print(t1-t0)
                print(logical_error_rate)
                if logical_error_rate < 0.1130:
                    break
        torch.save(van, abspath(dirname(__file__))+'/net/cir/d{}_r{}c1.pt'.format(d, r))
    elif Ctraining == False and decoding == True:
        trials = 50000
        path = abspath(dirname(__file__))+'/net/cir/d{}_r{}c1.pt'.format(d, r)
        net = torch.load(path)
        dtype=net.deep_net[0].weight.dtype
        van = MADE(n=ni, depth=args.depth, width=args.width).to(device).to(dtype)
        van.deep_net = net.deep_net.to(device).to(dtype)
        
        for i in range(10):
            lconf = forward(n_s=50000, m=ni-1, van=van, syndrome=input[:, :-1], device=device, dtype=dtype, k=1/2)
            logical_error_rate = abs(logicals-lconf).sum()/50000
            print(logical_error_rate)
        t0 = time.time()
        t = []
        for i in range(10):
            t2 = time.time()
            lconf = forward(n_s=trials, m=ni-1, van=van, syndrome=input[:trials, :-1].squeeze(0), device=device, dtype=dtype, k=1/2)
            logical_error_rate = abs(logicals[:trials]-lconf).sum()/trials
            t3 = time.time()
            print(t3-t2)
            t.append(t3-t2)
            print(logical_error_rate)
            t4 = time.time()
            print(t4-t3)
        t1 = time.time()
        import numpy as np
        print((t1-t0)/10, np.array(t).std())


    def count_logical_errors(detector_error_model, detection_events, observable_flips):
        from pymatching import Matching
        # Sample the circuit.
        num_shots = len(detection_events)
        matcher = Matching.from_detector_error_model(detector_error_model)

        # Run the decoder.
        
        # print(predictions)
        # print(observable_flips)
        # Count the mistakes.
        num_errors = 0
        t = np.zeros(10)
        for i in range(10):
            t0 = time.time()
            for shot in range(num_shots):
                predictions = matcher.decode(detection_events[shot]).squeeze()
                actual_for_shot = observable_flips[shot]
                predicted_for_shot = predictions
                if not np.array_equal(actual_for_shot, predicted_for_shot):
                    num_errors += 1
            t1 = time.time()
            t[i] = t1-t0
        print(t.mean(), t.std())
        return num_errors
    
    lmw = count_logical_errors(detector_error_model=dem, detection_events=data.syndromes()[:trials], observable_flips=data.logical_flip()[:trials])
