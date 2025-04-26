from args import args
import torch
import time
import sys
from pymatching import Matching
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Data, mod2


def count_logical_errors(detector_error_model, detection_events, observable_flips):
    # Sample the circuit.
        num_shots = detection_events.shape[0]
        matcher = Matching.from_detector_error_model(detector_error_model)

        # Run the decoder.
        predictions = matcher.decode_batch(detection_events).squeeze()

        # Count the mistakes.
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            # print(actual_for_shot, predicted_for_shot)
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return num_errors

def forward(n_s, m, van, syndrome, device, dtype, k=1):
    condition = syndrome*2-1
    x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    x = x[:, m:m+int(2*k)]
    return x

if __name__ == '__main__':
    from module import Surfacecode
    
    import stim
    import numpy as np
    import re
    
    d=7
    r=4
    k = 2
    er = 0.015
    trials = 1
    device='cuda:1'
    dtype=torch.float32
    
    cir_path = abspath(dirname(__file__)).strip('decoding')+'/module/circuit/d{}r{}_l{}.stim'.format(d, r, k-1)
    print(cir_path)

    netpath = abspath(dirname(__file__))+'/net/cir/'+'sim_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
    van = torch.load(netpath)
    van = van.to(device).to(dtype)
    
    print(van.depth)

    error_rates = torch.logspace(-3.2, -2, 10)
    print(error_rates)
    gnd = []
    mw = []
    # gndstd = []


    T=[]
    for j in range(100):
        lomw = []
        lognd = []
        for i in range(len(error_rates)):
            print(j, i)
            print(error_rates[i].item())
            with open(cir_path, 'r') as file:
                content = file.read()
                pattern = r'\(\d+\.\d+\)'
                new_content = re.sub(pattern, f'({error_rates[i]})', content)
            with open(cir_path, 'w') as file:
                file.write(new_content)

            defect_circuit = stim.Circuit.from_file(cir_path)

            sampler = defect_circuit.compile_sampler(seed=int(error_rates[i]*10000+j*10))
            meas = sampler.sample(shots=trials)
            # print(meas[0])
            dets0, obvs0 = defect_circuit.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
            ni = len(dets0[0])+len(obvs0[0])
            # print(len(dets[0]))
            # print(obvs[0])
            
            dem = defect_circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
            lmw = count_logical_errors(detector_error_model=dem, detection_events=dets0, observable_flips=obvs0*1)/trials

            # print(lmw)


            
            # van = MADE(n=ni, depth=4, width=35, residual=False).to(device).to(dtype)

            
            t0 = time.time()
            if trials == 1:
                syndrome = torch.tensor(dets0).squeeze()*1.0
                # print(syndrome)
            else:
                None
            # syndrome = torch.tensor(dets0)*1.0
            # print(syndrome)
            lconf = forward(n_s=trials, m=ni-k, van=van, syndrome=syndrome, device=device, dtype=dtype, k=k/2)
            t1 = time.time()
            print(t1-t0)
            T.append(t1-t0)
            # print(lconf[0])
            aclo = (torch.tensor(obvs0)*1.0).to(device).to(dtype)
            # print(aclo[0])
            logical_error_rate = torch.count_nonzero((aclo-lconf).sum(1))/trials
            
            lomw.append(lmw)
            lognd.append(logical_error_rate.cpu().item())
            print('mw:', lmw)
            print('gnd:', logical_error_rate.cpu().item())
        mw.append(lomw)
        gnd.append(lognd)

    print(van)
    print(van.depth,van.width)
    
    print('mw:', torch.tensor(mw).mean(0))
    print('gnd:', torch.tensor(gnd).mean(0))
    print('std:', torch.tensor(gnd).std(0))
    print('T:', torch.tensor(T).mean().item())
    # path = abspath(dirname(__file__))+'/net/cir/'+'sim_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
    # torch.save(van, path)
