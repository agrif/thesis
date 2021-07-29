import numpy
import scipy
import scipy.integrate
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.patches

def lorenz(t, y):
    sigma = 10
    beta = 8 / 3
    rho = 28
    dy0 = sigma * (y[1] - y[0])
    dy1 = y[0] * (rho - y[2]) - y[1]
    dy2 = y[0] * y[1] - beta * y[2]
    return [dy0, dy1, dy2]

lorenz_initial = [17.67715816276679, 12.931379185960404, 43.91404334248268]

def return_map(fn, i, initial, tmax, dt, method='RK23', rtol=1e-3, cmp=numpy.greater):
    ts = numpy.linspace(0, tmax, int(tmax / dt))
    sol = scipy.integrate.solve_ivp(fn, (0, tmax), initial, t_eval=ts, method=method, rtol=rtol)

    idx = scipy.signal.argrelextrema(sol.y[i], cmp)[0]
    ex = sol.y[i, idx]

    return numpy.stack([ex[:-1], ex[1:]], axis=-1)

def return_map_spline(fn, i, initial, tmax, dt, method='RK23', rtol=1e-3):
    ts = numpy.linspace(0, tmax, int(tmax / dt))
    sol = scipy.integrate.solve_ivp(fn, (0, tmax), initial, t_eval=ts, method=method, rtol=rtol)
    v = sol.y[i]

    spline = scipy.interpolate.InterpolatedUnivariateSpline(numpy.arange(len(v)), v, k=4)
    spline_d = spline.derivative()
    spline_dd = spline_d.derivative()
    extimes = spline_d.roots()

    # discard times out of bound
    extimes = extimes[extimes > 0]
    extimes = extimes[extimes < len(v) - 1]

    # select only local maxima
    extimes = extimes[spline_dd(extimes) < 0]

    # find values
    ex = spline(extimes)

    # construct return map
    return numpy.stack([ex[:-1], ex[1:]], axis=-1)

def plot(title, maps):
    print(title)
    zooms = [
        ((34.6, 35.5), (35.7, 36.6)),
    ]
    width = int(numpy.ceil(numpy.sqrt(1 + len(zooms))))
    height = int(numpy.ceil((1 + len(zooms)) / width))
    fig, axes = plt.subplots(height, width, dpi=200, figsize=(3 * width, 3 * height))
    axes = numpy.reshape(axes, (-1,), order='C')
    markers = ['P', 'X', 's', '^']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    #fig.suptitle(title)
    for (label, m), marker, color in zip(maps.items(), markers, colors):
        for i, ax in enumerate(axes[:1 + len(zooms)]):
            s = 2
            if i > 0:
                s = 5
            ax.scatter(m[:, 0], m[:, 1], label=label, s=s, linewidths=0, color=color, marker=marker)
    for i, ((ax, bx), (ay, by)) in enumerate(zooms):
        rect = matplotlib.patches.Rectangle((ax, ay), bx - ax, by - ay, linewidth=1, edgecolor='k', facecolor='none')
        axes[0].add_patch(rect)
        axes[i + 1].set_xlim(ax, bx)
        axes[i + 1].set_ylim(ay, by)
    #legend = axes[0].legend()
    #for h in legend.legendHandles:
    #    h._sizes = [10]
    axes[0].set_xlim(30, 48)
    axes[0].set_ylim(30, 48)
    for i, ax in enumerate(axes):
        ax.set_xlabel('$z_i$')
        ax.set_ylabel('$z_{i+1}$')
        sublabel = ['(a)', '(b)'][i]
        xpos = -0.1
        if i > 0:
            xpos = -0.25
        ax.text(xpos, 1.05, sublabel, transform=ax.transAxes, fontsize=10, va='top', ha='right')
    plt.tight_layout()
    #plt.savefig(title + '.png')
    plt.savefig('../figures/' + title + '.svg')
    plt.savefig('../figures/' + title + '.eps')
    #plt.show()

if __name__ == '__main__':
    # plot('RK23 Naive', {
    #    'dt = 0.02': return_map(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK23'),
    #    'dt = 0.01': return_map(lorenz, 2, lorenz_initial, 5000, 0.01, method='RK23'),
    #    'dt = 0.001': return_map(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK23'),
    # })

    # plot('RK45 Naive', {
    #     'dt = 0.02': return_map(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK45'),
    #     'dt = 0.01': return_map(lorenz, 2, lorenz_initial, 5000, 0.01, method='RK45'),
    #     'dt = 0.001': return_map(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK45'), 
    # })

    #plot('LSODA Naive', {
    #    'dt = 0.02': return_map(lorenz, 2, lorenz_initial, 5000, 0.02, method='LSODA'),
   #     'dt = 0.01': return_map(lorenz, 2, lorenz_initial, 5000, 0.01, method='LSODA'),
   #     'dt = 0.001': return_map(lorenz, 2, lorenz_initial, 5000, 0.001, method='LSODA'), 
    #})

    # plot('RK23 Spline', {
    #     'dt = 0.02': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK23'),
    #     'dt = 0.01': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.01, method='RK23'),
    #     'dt = 0.001': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK23'), 
    # })

    # plot('RK45 Spline', {
    #     'dt = 0.02': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK45'),
    #     'dt = 0.01': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.01, method='RK45'),
    #     'dt = 0.001': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK45'), 
    # })

    # plot('RK23 Compare', {
    #     'dt = 0.001, naive': return_map(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK23'),
    #     'dt = 0.02, spline': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK23'),
    # })

    # plot('RK45 Compare', {
    #     'dt = 0.001, naive': return_map(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK45'),
    #     'dt = 0.02, spline': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK45'),
    # })

    # plot('RK45 Compare (rtol = 1e-5)', {
    #     'dt = 0.001, naive': return_map(lorenz, 2, lorenz_initial, 5000, 0.001, method='RK45', rtol=1e-5),
    #     'dt = 0.02, spline': return_map_spline(lorenz, 2, lorenz_initial, 5000, 0.02, method='RK45', rtol=1e-5),
    # })

    plot('lorenz-rmap-tol', {
        'rk23': return_map_spline(lorenz, 2, lorenz_initial, 500, 0.025, method='RK23', rtol=1e-3),
        'rk45': return_map_spline(lorenz, 2, lorenz_initial, 500, 0.025, method='RK45', rtol=1e-3),
        'rk23-tol': return_map_spline(lorenz, 2, lorenz_initial, 500, 0.025, method='RK23', rtol=1e-5),
        'rk45-tol': return_map_spline(lorenz, 2, lorenz_initial, 500, 0.025, method='RK45', rtol=1e-5),
    })

