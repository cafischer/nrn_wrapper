import numpy as np
import matplotlib.pyplot as pl
from nrn_wrapper import Cell, iclamp_adaptive
import pandas as pd


data = pd.read_csv('/media/caro/Daten/Phd/DAP-Project/cell_fitting/data/2015_08_06d/raw/PP(4)/0(nA).csv')
#data = pd.read_csv('/media/caro/Daten/Phd/DAP-Project/cell_fitting/data/2015_08_06d/raw/IV/-0.1(nA).csv')
t = data.t.values
dt = t[1]-t[0]
tstop = t[-1]
i_inj = data.i.values
v_init = data.v.values[0]
sec = ['soma', None]
start_step = int(round(222 / dt))
end_step = start_step + int(round(250 / dt))
#start_step = int(round(250 / dt))
#end_step = int(round(750 / dt))
discontinuities = [start_step, end_step]
print dt

"""
dt = 0.01
tstop = 100
t = np.arange(0, tstop+dt, dt)
i_inj = np.zeros(len(t))
i_inj[np.logical_and(20 <= t, t < 60)] = 1
v_init = -60
sec = ['soma', None]
start_idx = int(np.round(20.0/dt))  #np.where(np.abs(np.diff(20 <= t)))[0] + 1
end_idx = int(np.round(60.0/dt))  #np.where(np.abs(np.diff(t <= 60)))[0]
discontinuities=[start_idx, end_idx]
"""

# create cell
cell = Cell.from_modeldir('../demo/test_cell.json', '../demo/channels')


v_rec, t_rec, i_rec = iclamp_adaptive(cell, sec, i_inj, v_init, tstop, dt, celsius=35, pos_i=0.5, pos_v=0.5, atol=1e-8,
                                      continuous=True, discontinuities=discontinuities, interpolate=True)

print tstop
print t_rec[-1]
print np.diff(t_rec)[1:10]

pl.figure()
pl.plot(t_rec, v_rec)
pl.show()

pl.figure()
pl.plot(t, i_inj, 'x')
pl.plot(t_rec, i_rec)
pl.show()