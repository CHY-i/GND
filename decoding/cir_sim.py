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
    trials = 10000
    cotrain = False
    device = 'cuda:1'

    cir_path = abspath(dirname(__file__)).strip('decoding')+'/module/circuit/d{}r{}_l{}.stim'.format(d, r, k-1)
    print(cir_path)
    
    with open(cir_path, 'r') as file:
        content = file.read()
        pattern = r'\(\d+\.\d+\)'
        new_content = re.sub(pattern, f'({er})', content)
    with open(cir_path, 'w') as file:
        file.write(new_content)

    defect_circuit = stim.Circuit.from_file(cir_path)

    sampler = defect_circuit.compile_sampler(seed=97124)
    meas = sampler.sample(shots=trials)
    # print(meas[0])
    dets0, obvs0 = defect_circuit.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    # print(len(dets[0]))
    # print(obvs[0])

    

    
    dem = defect_circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
    lmw = count_logical_errors(detector_error_model=dem, detection_events=dets0, observable_flips=obvs0*1)/trials

    print(lmw)

    

    if cotrain == False:
        epoch = 150000
        lr = 0.001
        batch  = 50000
        dtype = torch.float32
        ni = len(dets0[0])+len(obvs0[0])
        print(ni)
        van = MADE(n=ni, depth=4, width=30, residual=False).to(device).to(dtype)

        optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
        # loss_his = []
        # lo_his = []
        for l in range(epoch):
            
            # sampler = defect_circuit.compile_sampler(seed=7453+epoch)
            meas = sampler.sample(shots=batch)
            dets, obvs = defect_circuit.compile_m2d_converter().convert(measurements=meas, separate_observables=True)

            s = torch.hstack((torch.tensor(dets)*1.0, torch.tensor(obvs)*1.0)).to(device).to(dtype)
            logp = van.log_prob((s*2-1))
        
            loss = torch.mean((-logp), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
                scheduler.step()
            
            if (l) % 1000 == 0:
                print('epoch:', l)
                t0 = time.time()
                lconf = forward(n_s=10000, m=ni-k, van=van, syndrome=torch.tensor(dets0)*1.0, device=device, dtype=dtype, k=k/2)
                # print(lconf[0])
                aclo = (torch.tensor(obvs0)*1.0).to(device).to(dtype)
                # print(aclo[0])
                logical_error_rate = torch.count_nonzero((aclo-lconf).sum(1))/10000
                t1 = time.time()
                print(t1-t0)
                print('mw:', lmw)
                print('gnd:', logical_error_rate)

        path = abspath(dirname(__file__))+'/net/cir/'+'sim_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
        torch.save(van, path)
    
    elif cotrain == True:
        epoch = 50000
        lr = 0.0001
        batch  = 50000

        dtype = torch.float32
        ni = len(dets0[0])+len(obvs0[0])

        netpath = abspath(dirname(__file__))+'/net/cir/'+'c_sim_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
        van = torch.load(netpath)
        van = van.to(device).to(dtype)

        optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
        # scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
        # loss_his = []
        # lo_his = []
        for l in range(epoch):
            # print(l)
            # sampler = defect_circuit.compile_sampler(seed=7453+epoch)
            meas = sampler.sample(shots=batch)
            dets, obvs = defect_circuit.compile_m2d_converter().convert(measurements=meas, separate_observables=True)

            s = torch.hstack((torch.tensor(dets)*1.0, torch.tensor(obvs)*1.0)).to(device).to(dtype)
            logp = van.log_prob((s*2-1))
        
            loss = torch.mean((-logp), dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
            #     scheduler.step()
            
            if (l) % 1000 == 0:
                print('epoch:', l)
                t0 = time.time()
                lconf = forward(n_s=10000, m=ni-k, van=van, syndrome=torch.tensor(dets0)*1.0, device=device, dtype=dtype, k=k/2)
                # print(lconf[0])
                aclo = (torch.tensor(obvs0)*1.0).to(device).to(dtype)
                # print(aclo[0])
                logical_error_rate = torch.count_nonzero((aclo-lconf).sum(1))/10000
                t1 = time.time()
                print(t1-t0)
                print('mw:', lmw)
                print('gnd:', logical_error_rate)

        path = abspath(dirname(__file__))+'/net/cir/'+'c_sim_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
        torch.save(van, path)
