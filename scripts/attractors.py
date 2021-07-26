import numpy
import scipy.integrate
import jitcdde
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Times:
  def __init__(self, lyapunov, dt):
    self.dt = dt
    self.lyapunov = lyapunov
    self.lyapunov_inv = 1.0 / lyapunov
    
    self.transient_time = 10 #/ lyapunov
    self.train_time = 10 #/ lyapunov
    self.predict_time = 100 #/ lyapunov
    
    self.train_display = 1 #/ lyapunov
    self.predict_display = 5 #/ lyapunov
    
    self.warmup_time = 3 #/ lyapunov
    self.nrmse_time = 1 / lyapunov
    self.total_trial_time = self.warmup_time + self.nrmse_time
    
    self.transient_start = 0
    self.transient_end = self.transient_start + self.transient_time
    
    self.train_start = self.transient_end
    self.train_end = self.train_start + self.train_time
    
    self.predict_start = self.train_end
    self.predict_end = self.predict_start + self.predict_time
  
  def i(self, t):
    return int(numpy.round(t / self.dt))

class Input:
  def __init__(self, times, ts, ys):
    self.times = times
    self.ts = ts
    self.ys = ys
    
    assert len(ts.shape) == 1
    assert len(ys.shape) == 2
    assert ts.shape[0] == ys.shape[0]
    self.dimension = ys.shape[1]
  
  def eval(self, t):
    return numpy.array([numpy.interp(t, self.ts, self.ys[:,i]) for i in range(self.dimension)]).T
  
  def plot(self):
    plt.figure(figsize=(5, 3), dpi=200)
    i = self.times.i(self.times.lyapunov_inv * 10)
    plt.plot(self.ts[:i], self.ys[:i])
    plt.ylabel('u(t)')
    plt.xlabel('t')
  
  def phase(self):
    f, ax = plt.subplots(figsize=(3, 3), dpi=200)
    ts = numpy.arange(self.times.predict_start, self.times.predict_end, self.times.dt)
    self.phase_part(ax, ts, self.eval(ts))
  
  def break_symmetry(self, rs):
    return rs

class Lorenz63(Input):
  sigma = 10
  beta = 8 / 3
  rho = 28
  
  lyapunov = 0.9056
  dt = 0.01
  
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    times = Times(self.lyapunov, self.dt)
    
    def lorenz(t, y):
      dy0 = self.sigma * (y[1] - y[0])
      dy1 = y[0] * (self.rho - y[2]) - y[1]
      dy2 = y[0] * y[1] - self.beta * y[2]
      return [dy0, dy1, dy2]
    
    ts = numpy.arange(0, times.predict_end, times.dt)
    sol = scipy.integrate.solve_ivp(lorenz, (0, ts[-1]), [0.89660706, 1.6297056, 9.90638289], t_eval=ts, method='RK45')
    
    ys = sol.y.T
    #ys -= numpy.mean(ys, axis=0)
    #ys /= numpy.std(ys, axis=0)
    
    super().__init__(times, ts, ys)
  
  def phase_part(self, ax, ts, ys, times=[]):
    lw = 20.0 / self.times.predict_time
    
    ax.set_ylabel('$z(t)$')
    ax.set_xlabel('$x(t)$')
    ax.set_xticks(ticks=[], minor=[])
    ax.set_yticks(ticks=[], minor=[])
    
    points = numpy.array([self.eval(t)[0::2] for t in times])
    if len(points):
      ax.scatter(points[:, 0], points[:, 1], c='red', zorder=2)
    ax.plot(ys[:, 0], ys[:, 2], lw=lw, color='black', zorder=1)
  
  def break_symmetry(self, rs):
    last_half = int(numpy.round(rs.shape[1] / 2))
    rs = numpy.copy(rs)
    rs[:, last_half:] **= 2
    return rs

class MackeyGlass(Input):
  beta = 0.2
  gamma = 0.1
  tau = 17
  n = 10
  
  # !!! this is rescaled in init to match Lorenz63
  delay = tau
  lyapunov = 0.0086
  dt = 0.01
  
  def __init__(self):
    # !!! lyapunov rescaling happens here
    self.timescale = Lorenz63.lyapunov / self.lyapunov
    self.lyapunov = Lorenz63.lyapunov

    times = Times(self.lyapunov, self.dt)
    
    dde = jitcdde.jitcdde([self.timescale * self.beta * jitcdde.y(0, jitcdde.t - self.tau / self.timescale) / (1 + jitcdde.y(0, jitcdde.t - self.tau / self.timescale)**self.n) - self.timescale * self.gamma * jitcdde.y(0)])
    dde.constant_past([0.5])
    dde.step_on_discontinuities()
    
    ts = numpy.arange(dde.t, dde.t + 5 * 2000, times.dt)
    ys = numpy.zeros((len(ts), 1))
    for i, t in enumerate(ts):
      ys[i] = dde.integrate(t)
    start = dde.t
    
    ts = numpy.arange(dde.t, dde.t + times.predict_end + 200, times.dt)
    ys = numpy.zeros((len(ts), 1))
    for i, t in enumerate(ts):
      ys[i] = dde.integrate(t)
    ts -= start
    
    #ys = numpy.tanh(ys - 1)
    #ys /= numpy.max(numpy.abs(ys))
    
    super().__init__(times, ts, ys)
  
  def phase_part(self, ax, ts, ys, times=[]):
    lag_i = self.times.i(self.tau / self.timescale)
    lw = 10.0 / self.times.predict_time
    
    ax.set_xlim(-1.1, 0.7)
    ax.set_ylim(-1.1, 0.7)
    ax.set_xlabel('$u(t)$')
    ax.set_ylabel('$u(t - {})$'.format(self.tau))
    ax.set_xticks(ticks=[], minor=[])
    ax.set_yticks(ticks=[], minor=[])

    ax.plot(ys[lag_i:], ys[:-lag_i], lw=lw, color='black')
    
    points = numpy.array([(self.eval(t + self.tau), self.eval(t)) for t in times])
    if len(points):
      ax.scatter(points[:, 0], points[:, 1], c='red')

# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.77.1751
class DoubleScroll(Input):
  R1 = 1.2
  R2 = 3.44
  R4 = 0.193
  Ir = 2.25 * 10**-5
  alpha = 11.6
  
  # !!! this is rescaled in init to match Lorenz63
  # from Dan's email and data
  lyapunov = 0.091
  dt = 0.01
  
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    
    # !!! lyapunov rescaling happens here
    self.timescale = Lorenz63.lyapunov / self.lyapunov
    self.lyapunov = Lorenz63.lyapunov
    
    times = Times(self.lyapunov, self.dt)
    
    def g(V):
      return V / self.R2 + 2 * self.Ir * numpy.sinh(self.alpha * V)
    
    def double_scroll(t, y):
      V1, V2, I = y
      dV1 = V1 / self.R1 - g(V1 - V2)
      dV2 = g(V1 - V2) - I
      dI = V2 - self.R4 * I
      return [dV1 * self.timescale, dV2 * self.timescale, dI * self.timescale]
    
    ts = numpy.arange(0, times.predict_end, times.dt)
    sol = scipy.integrate.solve_ivp(double_scroll, (0, ts[-1]), [-1.10540933, -0.24504292, -0.5758393], t_eval=ts, method='RK45')
    
    ys = sol.y.T
    #ys -= numpy.mean(ys, axis=0)
    #ys /= numpy.std(ys, axis=0)
    
    super().__init__(times, ts, ys)
  
  def phase_part(self, ax, ts, ys, times=[]):
    lw = 20.0 / self.times.predict_time
    
    ax.set_ylabel('$I(t)$')
    ax.set_xlabel('$V_1(t)$')
    ax.set_xticks(ticks=[], minor=[])
    ax.set_yticks(ticks=[], minor=[])
    
    points = numpy.array([self.eval(t)[0::2] for t in times])
    if len(points):
      ax.scatter(points[:, 0], points[:, 1], c='red', zorder=2)
    ax.plot(ys[:, 0], ys[:, 2], lw=lw, color='black', zorder=1)
  
  def break_symmetry(self, rs):
    last_half = int(numpy.round(rs.shape[1] / 2))
    rs = numpy.copy(rs)
    rs[:, last_half:] **= 2
    return rs

# https://www.sciencedirect.com/science/article/pii/0375960176901018?via%3Dihub
class Rossler(Input):
  a = 0.2
  b = 0.2
  c = 5.7
  
  # !!! this is rescaled in init to match Lorenz63
  lyapunov = 0.0714
  dt = 0.002
  
  # apply log to z component?
  use_log_z = True
  
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    
    # !!! lyapunov rescaling happens here
    self.timescale = Lorenz63.lyapunov / self.lyapunov
    self.lyapunov = Lorenz63.lyapunov
    
    times = Times(self.lyapunov, self.dt)
    
    def rossler(t, y):
      dx = -y[1] - y[2]
      dy = y[0] + self.a * y[1]
      dz = self.b + y[2] * (y[0] - self.c)
      return [dx * self.timescale, dy * self.timescale, dz * self.timescale]
    
    ts = numpy.arange(0, times.predict_end, times.dt)
    sol = scipy.integrate.solve_ivp(rossler, (0, ts[-1]), [2.56922952, 6.78429634, 0.93323071], t_eval=ts, method='RK45')
    
    ys = sol.y.T
    #if self.use_log_z:
    #  ys[:, 2] = numpy.log(ys[:, 2])
    #self.oldmean = numpy.mean(ys, axis=0)
    #ys -= self.oldmean
    #self.oldstd = numpy.std(ys, axis=0)
    #ys /= self.oldstd
    
    super().__init__(times, ts, ys)
  
  def phase_part(self, ax, ts, ys, times=[]):
    lw = 20.0 / self.times.predict_time
    
    ax.set_ylabel('$z(t)$')
    ax.set_xlabel('$x(t)$')
    ax.set_xticks(ticks=[], minor=[])
    ax.set_yticks(ticks=[], minor=[])
    
    ys = ys.copy()
    ys *= self.oldstd
    ys += self.oldmean
    if self.use_log_z:
      ys[:, 2] = numpy.exp(ys[:, 2])
    
    # fixme rescale
    points = numpy.array([self.eval(t)[0::2] for t in times])
    if len(points):
      ax.scatter(points[:, 0], points[:, 1], c='red', zorder=2)
    ax.plot(ys[:, 0], ys[:, 2], lw=lw, color='black', zorder=1)
  
  def break_symmetry(self, rs):
    last_half = int(numpy.round(rs.shape[1] / 2))
    rs = numpy.copy(rs)
    rs[:, last_half:] **= 2
    return rs

def plot_2d(xs, ys, xlabel, ylabel, outname, lw=0.1, skip=0):
    plt.close('all')
    f, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(xs[skip:], ys[skip:], lw=lw, color='black')
    plt.tight_layout()
    plt.savefig('../figures/' + outname)
    plt.savefig('../figures/' + outname.replace('.svg', '.eps'))
    #plt.show()

def plot_3d(data, xlabel, ylabel, zlabel, outname, lw=0.1, elev=None, azim=None):
    plt.close('all')
    f = plt.figure(figsize=(4, 4), dpi=200)
    ax = plt.gca(projection='3d')
    ax.dist = 13
    if elev is not None:
        ax.elev = elev
    if azim is not None:
        ax.azim = azim
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.plot3D(data[:, 0], data[:, 1], data[:, 2], lw=lw, color='black')
    plt.tight_layout()
    plt.savefig('../figures/' + outname)
    plt.savefig('../figures/' + outname.replace('.svg', '.eps'))
    #plt.show()

l63 = Lorenz63()
plot_2d(l63.ys[:, 0], l63.ys[:, 2], '$x(t)$', '$z(t)$', 'lorenz.svg')
plot_3d(l63.ys, '$x(t)$', '$y(t)$', '$z(t)$', 'lorenz-3d.svg')
#print('lorenz', l63.ys[-1])

ross = Rossler()
plot_2d(ross.ys[:, 0], ross.ys[:, 2], '$x(t)$', '$z(t)$', 'rossler.svg')
plot_3d(ross.ys, '$x(t)$', '$y(t)$', '$z(t)$', 'rossler-3d.svg')
#print('rossler', ross.ys[-1])

dscroll = DoubleScroll()
plot_2d(dscroll.ys[:, 0], dscroll.ys[:, 2], '$V_1(t)$', '$I(t)$', 'dscroll.svg')
plot_3d(dscroll.ys, '$V_1(t)$', '$V_2(t)$', '$I(t)$', 'dscroll-3d.svg', azim=-30, elev=15)
#print('dscroll', dscroll.ys[-1])

mg = MackeyGlass()
mg_lag_i = mg.times.i(mg.tau / mg.timescale)
plot_2d(mg.ys[mg_lag_i:, 0], mg.ys[:-mg_lag_i, 0], '$u(t)$', '$u(t - \\tau)$', 'mackey-glass.svg', skip=22000)
#print('mackey-glass', mg.ys[-1])
