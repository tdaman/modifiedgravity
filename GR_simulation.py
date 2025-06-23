import numpy as np

from modified_GR_simulation import plot_state
from scipy.integrate import solve_ivp

def GW_derivatives(a,state,k):
    h, h1 = state
    H = 1/a # note that density0 is always 3 in natural units for flat GR universe.

    h2 = -2 * H * h1 - k ** 2 * h

    frac_aH = 1 / (a * H)
    h1_derivative = frac_aH * h2
    h_derivative = frac_aH * h1
    return h_derivative, h1_derivative

def energy_analysis(k, scales, GW, GW_derivative):
    # note that the initial energy depends on k (=k**2), so we must scale it to find the damping spectrum
    final_energy = k ** 2 * GW[-1] ** 2 + GW_derivative[-1] ** 2
    return final_energy / k ** 2

def GW_simulator(scale_range,k_range,analysis_function,plot_title,output_points=1000,plot_strain=False):
    # start with flat spectrum, so all h_ij(k),a=1)=1 and zero velocity
    h_init = 1
    h1_init = 0
    init_state = [h_init, h1_init]
    t_eval = np.logspace(np.log10(scale_range[0]), np.log10(scale_range[1]) - 0.000000000001, 1000)

    damping_spectrum = []

    for k in k_range:
        print(f"\rk={k:.3g}",end="")
        args = [k]
        output = solve_ivp(GW_derivatives, scale_range, init_state, args=args, t_eval=t_eval,atol=1e-9, rtol=1e-8, method='DOP853')
        scales, states = output['t'], output['y']
        GW = states[0]
        GW_derivative = states[1]
        state_names = ["GW strain", "GW derivative"]
        if plot_strain:
            for state_name, state_points in zip(state_names, [GW, GW_derivative]):
                plot_state(scales, state_points, state_name, plot_title + f",k={k:2g}", k)
        damping = analysis_function(k, scales, GW, GW_derivative)
        damping_spectrum.append(damping)
    print()
    return np.array(damping_spectrum)

def simulation(final_scale, analysis_function=energy_analysis, plot_strain=False,k_low=None,k_high=0,k_range=None):
    print("Starting GR simulation")
    init_scale = 1
    scale_range = [init_scale, final_scale]
    # This is the radiation energy density. Determined by flat universe condition
    plot_title = "GR"
    if k_range is None:
        k_amount = 200 if not plot_strain else 2
        if k_low is None:
            k_low = -np.log10(final_scale) if not plot_strain else -1
        k_range = np.logspace(k_low, k_high, k_amount)
    return k_range,GW_simulator(scale_range, k_range, analysis_function, plot_title,plot_strain=plot_strain)

def main():
    def dummy_analysis_function(k,scales,GW,GW_derivative):
        return None
    simulation(final_scale=20,analysis_function=dummy_analysis_function,plot_strain=True)

if __name__ == "__main__":
    main()