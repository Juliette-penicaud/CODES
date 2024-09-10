import numpy as np

# First signal
sig1 = np.sin(np.r_[-1:1:0.1])
# Seconds signal with pi/4 phase shift. Half the size of sig1
sig2 = np.sin(np.r_[-1:0:0.1] + np.pi/4)
print(sig1)
print('sig2', sig2)

corr = np.correlate(a=sig1, v=sig2)
print(corr)