import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotter(margin_function, args=(), long_call_positions=np.arange(0, 500, 25), short_call_positions=np.arange(-1000, 0, 25)):
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(long_call_positions, short_call_positions)

    Z = np.array([-margin_function(np.array([float(x), float(y), 500]), *args) 
                  for xr, yr in zip(X, Y) 
                      for x, y in zip(xr,yr) ]
                 ).reshape(len(X), len(X[0]))

    N = Z / Z.max()  # normalize 0 -> 1 for the colormap
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1, facecolors=cm.jet(N))

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(Z)
    plt.colorbar(m, shrink=0.8, aspect=20)
    ax.view_init(30, 330)

    plt.show()


def d_plus_minus(strike, tte, spot, vol, r, q):
    vol_root_t = vol * np.sqrt(tte)
    d_plus = (np.log(spot/strike) + (r - q) * tte + 0.5 * vol_root_t**2) / vol_root_t
    d_minus = d_plus - vol_root_t
    return d_plus, d_minus, vol_root_t


def spot_price(strike, tte, spot, vol, r, q, cp=1):
    d_plus, d_minus, vol_root_t = d_plus_minus(strike, tte, spot, vol, r, q)
    return cp * (spot * np.exp(-q * tte) * norm.cdf(cp * d_plus) - strike * np.exp(-r * tte) * norm.cdf(cp * d_minus))


def relative_delta(strike, tte, spot, vol, r, q, cp=1):
    d_plus, d_minus, vol_root_t = d_plus_minus(strike, tte, spot, vol, r, q)
    return spot * np.exp(-q * tte) * (norm.cdf(d_plus) - 0.5 * (1 - cp))


def relative_vega(strike, tte, spot, vol, r, q, cp=1):
    d_plus, d_minus, vol_root_t = d_plus_minus(strike, tte, spot, vol, r, q)
    return spot * np.exp(-q * tte) * vol_root_t * norm.pdf(d_plus)


def relative_gamma(strike, tte, spot, vol, r, q, cp=1):
    d_plus, d_minus, vol_root_t = d_plus_minus(strike, tte, spot, vol, r, q)
    return spot * np.exp(-q * tte) * norm.pdf(d_plus) / vol_root_t


def relative_vanna(strike, tte, spot, vol, r, q, cp=1):
    d_plus, d_minus, vol_root_t = d_plus_minus(strike, tte, spot, vol, r, q)
    return -spot * np.exp(-q * tte) * d_minus * norm.pdf(d_plus)


def relative_volga(strike, tte, spot, vol, r, q, cp=1):
    d_plus, d_minus, vol_root_t = d_plus_minus(strike, tte, spot, vol, r, q)
    return spot * np.exp(-q * tte) * vol_root_t * d_plus * d_minus * norm.pdf(d_plus)
