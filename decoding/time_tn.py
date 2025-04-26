import numpy as np
from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder
import time
# initialise models
my_code = RotatedPlanarCode(11, 11)
my_error_model = DepolarizingErrorModel()
my_decoder = RotatedPlanarMPSDecoder(chi=8)


trials = 10
error_probability = 0.189
# seed random number generator for repeatability
rng = np.random.default_rng(10)
tt = np.zeros((10))
for i in range (10):
    t = 0
    for j in range(trials):
        error = my_error_model.generate(my_code, error_probability, rng)
        syndrome = pt.bsp(error, my_code.stabilizers.T)
        t0 = time.time()
        recovery = my_decoder.decode(my_code, syndrome)
        a = pt.bsp(recovery ^ error, my_code.logicals.T)
        t1 = time.time()
        t+=t1-t0
    print(t)
    tt[i] = t
print(tt.mean(), tt.std(), tt.mean()/trials)
