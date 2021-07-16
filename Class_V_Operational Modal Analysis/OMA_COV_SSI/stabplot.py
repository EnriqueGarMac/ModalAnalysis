"""
##############################################
Visualization module
##############################################
All functions related to the plotting of results from OMA methods.
"""
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def stabplot(lambdtot, orderstot, lambd, orders, fn, phi=None, model=None, freq_range=None, frequency_unit='rad/s', damped_freq=False, psd_freq=None, psd_y=None, psd_plot_scale='log', 
    pole_settings=None, selected_pole_settings=None, to_clipboard='none', return_ix=False):
    """
    Generate plotly-based stabilization plot from output from find_stable_poles.

    Arguments
    ---------------------------
    lambd : double
        array with complex-valued eigenvalues
    orders : int
        corresponding order for each pole in `lambd`
    phi : optional, double
        matrix where each column is complex-valued eigenvector corresponding to lambd
    model : optional, double
        model object which is required input for plotting phi (based on geometry definition of system, constructed by `Model` class)
    freq_range : double, optional
        list of min and max values used for frequency axis
    frequency_unit : {'rad/s', 'Hz', 's'}, optional
        what frequency unit to use ('s' or 'period' enforces period rather than frequency) 
    damped_freq : False, optional
        whether or not to use damped frequency (or period) values in plot (False enforces undamped freqs)
    psd_freq : double, optional
        [not yet implemented] frequency values of plot to overlay, typically spectrum of data
    psd_y : double, optional
        [not yet implemented] function values of plot to overlay, typically spectrum of data
    psd_plot_scale: {'log', 'linear'}, optional
        how to plot the overlaid PSD (linear or logarithmic y-scale)
    renderer : 'browser_legacy', optional
        how to plot figure, refer plotly documentation for details 
        ('svg', 'browser', 'notebook', 'notebook_connected', are examples - 
        use 'default' to give default and None to avoid plot)
    to_clipboard : {'df', 'ix', 'none'}, optional
        update clipboard every time a pole is added, keeping selected indices or table
        'df' is not operational yet
    return_ix : False, optional
        whether or not to return second variable with indices - this is updated as more poles are selected
        

    Returns
    ---------------------------
    fig : obj
        plotly figure object

        
    Notes
    ----------------------------
    By hovering a point, the following data about the point will be given in tooltip:
        
         * Natural frequency / period in specified unit (damped or undamped)
         * Order 
         * Critical damping ratio in % (xi)
         * Index of pole (corresponding to inputs lambda_stab and order_stab)
    """

    fig, ax = plt.subplots(figsize=(20,10))
    # Twin the x-axis twice to make independent y-axes.
    axes = [ax, ax.twinx()]

    
    # Create suffix and frequency value depending on whether damped freq. is requested or not
    if damped_freq:
        dampedornot = 'd'
        omega = np.abs(np.imag(lambd))
    else:
        dampedornot = 'n'
        omega = np.abs(lambd)

    # Create frequency/period axis and corresponding labels
    if frequency_unit == 'rad/s':
        x = omega
        psd_freq = psd_freq*2*np.pi
        xlabel = f'$\omega_{dampedornot} \; [{frequency_unit}]$'
        tooltip_name = f'\omega_{dampedornot}'
        frequency_unit = 'rad/s'
    elif frequency_unit.lower() == 'hz':
        x = omega/(2*np.pi)
        xlabel = f'$f_{dampedornot} \; [{frequency_unit}]$'
        tooltip_name = f'f_{dampedornot}'
        frequency_unit = 'Hz'
    elif (frequency_unit.lower() == 's') or (frequency_unit.lower() == 'period'):
        x = (2*np.pi)/omega
        xlabel = f'Period, $T_{dampedornot} \; [{frequency_unit}]$'
        tooltip_name = f'T_{dampedornot}'
        frequency_unit = 's'
    
    # Damping ratio and index to hover
    xi = -np.real(lambd)/np.abs(lambd)

    # Construct dataframe and create scatter trace     
    poles = pd.DataFrame({'freq': x, 'order':orders})
    
    cont = 0
    for i in lambdtot:
        orderplot = orderstot[cont]
        cont = cont+1
        if damped_freq:
            freqplot = np.abs(np.imag(i))
        else:
            freqplot = np.abs(i)
        if frequency_unit == 'Hz':
            freqplot = freqplot/(2*np.pi)
        elif (frequency_unit.lower() == 's') or (frequency_unit.lower() == 'period'):   
            freqplot = (2*np.pi)/freqplot
        if cont == 1:
           axes[0].scatter(x=freqplot, y=np.repeat(orderplot,len(freqplot)), s=30, c='grey', zorder=1, label='Unstable pole') 
        else:
           axes[0].scatter(x=freqplot, y=np.repeat(orderplot,len(freqplot)), s=30, c='grey', zorder=1)
    axes[0].scatter(x=poles['freq'], y=poles['order'], s=30, c='r', zorder=2, label='Stable pole')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=2, fancybox=True, shadow=True, fontsize=15)
    for i in fn:
         axes[0].plot([i,i],[np.min(orderstot),np.max(orderstot)], '--r')
    axes[0].set_xlim(freq_range[0],freq_range[1])
    axes[0].set_ylim(np.min(orderstot),np.max(orderstot))
    axes[0].set_ylabel('Model order', fontsize=20)
    axes[0].set_xlabel('Frequency [Hz]', fontsize=20)
    
    # PSD overlay trace
    axes[1].plot(psd_freq, 10*np.log10(psd_y))
    axes[1].set_xlim(freq_range[0],freq_range[1])
    axes[1].yaxis.set_visible(False)