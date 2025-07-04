
import stim
from pymatching import Matching
import torch
import numpy as np
import time
import sys
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))

from module import MADE, TraDE_binary, NADE,  Errormodel, mod2, Loading_code, read_code, Abstractcode


def count_logical_errors(detector_error_model, detection_events, observable_flips):
    # Sample the circuit.
    num_shots = detection_events.shape[0]
    matcher = Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events).squeeze()
    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot][0]
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


d, r, k = 17, 1, 1
er = 0.12
trials = 100000
path = abspath(dirname(__file__))+'/net/cir/'+'rep_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
print(path)
'''generate circuit'''
circuit = stim.Circuit.generated(code_task="repetition_code:memory",
                                        distance=d,
                                        rounds=r,
                                        after_clifford_depolarization=er,
                                        before_measure_flip_probability=er,
                                        after_reset_flip_probability=er,
                                        )
'''define the detector error model and sampler'''
dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)

sampler = circuit.compile_detector_sampler(seed=None)
dets0, obvs0 = sampler.sample(shots=trials, separate_observables=True)
lmw = count_logical_errors(detector_error_model=dem, detection_events=dets0, observable_flips=obvs0*1)/trials # matching decoding results

epoch = 200000
lr = 0.001
batch  = 100000
dtype = torch.float32
device = 'cuda:3'
ni = len(dets0[0])+len(obvs0[0])
print(ni)
van = MADE(n=ni, depth=4, width=20, residual=False).to(device).to(dtype)#torch.load(path).to(device).to(dtype)#

optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
# loss_his = []
# lo_his = []
for l in range(epoch):
    
    sampler = circuit.compile_sampler(seed=8143+13*l)
    dets, obvs = sampler.sample(shots=batch, separate_observables=True)

    s = torch.hstack((torch.tensor(dets)*1.0, torch.tensor(obvs)*1.0)).to(device).to(dtype)
    logp = van.log_prob((s*2-1))

    loss = torch.mean((-logp), dim=0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
        scheduler.step()
    '''decoding test'''
    if (l) % 1000 == 0:
        print(loss)
        print('epoch:', l)
        t0 = time.time()
        lconf = forward(n_s=trials, m=ni-k, van=van, syndrome=torch.tensor(dets0)*1.0, device=device, dtype=dtype, k=k/2)
        # print(lconf[0])
        aclo = (torch.tensor(obvs0)*1.0).to(device).to(dtype)
        # print(aclo[0])
        logical_error_rate = torch.count_nonzero((aclo-lconf).sum(1))/trials
        t1 = time.time()
        print(t1-t0)
        print('mw:', lmw)
        print('gnd:', logical_error_rate)

# path = abspath(dirname(__file__))+'/net/cir/'+'rep_d{}_r{}_k{}_er{}.pt'.format(d, r, k, er)
# print(path)
torch.save(van, path)

