import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import modified_GR_model as Model

def set_params(params):
    Model.V0,Model.zeta,Model.xi,Model.density0 = params

def plot_state(scale_array,state_array,state_name,title,k=None,log=False,save_plot=False):
    plt.figure()
    plt.xlabel("Scale")
    plt.ylabel(state_name)
    plt.title(title)
    if log:
        plt.loglog(scale_array,state_array)
    else:
        plt.semilogx(scale_array,state_array)
    parameter_string = f"{Model.density0:.2g},{Model.xi:.2g},{Model.zeta:.2g},{Model.V0:.1g},a={int(scale_array[-1])}"
    if save_plot:
        if k is not None:
            filename = f"{state_name},{parameter_string},k={k:.1g}.pdf"
        else :
            filename = f"{state_name},{parameter_string}.pdf"
        plt.savefig(filename)
    plt.show()
    if state_name=="Scalar field":
        peaks,properties = find_peaks(state_array)
        peak_scales = scale_array[peaks]
        periods = np.diff(peak_scales)
        plt.figure()
        plt.plot(peak_scales[1:],periods)
        plt.title("Period analysis of the scalar field")
        plt.ylabel("Period")
        plt.xlabel("Scale")
        plt.show()


def quadratic(a,b,c):
    # the positive root solution of the quadratic equation
    return (-b+np.sqrt(b**2-4*a*c))/(2*a)

def invert_system(a,b,c,d,e,f):
    det = b*d-a*e
    adj_factor = a*f-c*d
    return adj_factor/det

def calc_Hubble(a,phi,psi):
    f = Model.f(phi)
    f1 = Model.f_derivative(phi)
    V = Model.potential(phi)
    g1 = Model.g_derivative(psi, a)
    rho = Model.density(a)
    X = Model.X(psi, a)

    # solve for H: quadratic equation
    c2 = 3 * f
    c1 = 3 * (f1 * psi + 2 * g1 * X * psi)
    c0 = -0.5 * psi ** 2 - a ** 2 * V - a ** 2 * rho
    return quadratic(c2, c1, c0)

def derivatives(a,state):
    phi,psi = state

    f = Model.f(phi)
    f1 = Model.f_derivative(phi)
    f2 = Model.f_derivative2(phi)
    V = Model.potential(phi)
    V1 = Model.potential_derivative(phi)
    # g = Model.g(psi, a)
    g1 = Model.g_derivative(psi, a)
    g2 = Model.g_derivative2(psi, a)
    rho = Model.density(a)
    P = Model.pressure(a)
    X = Model.X(psi,a)

    # solve for H: quadratic equation
    c2 = 3*f
    c1 = 3*(f1*psi+2*g1*X*psi)
    c0 = -0.5*psi**2-a**2*V-a**2*rho
    H = quadratic(c2,c1,c0)

    # solve for psi': linear system a1H'+a2psi'+a3=0,b1H'+b2psi'+b3=0
    a1 = 2*f
    a2 = f1+2*g1*X
    a3 = f*H**2 + f2*psi**2 + H*f1*psi + 0.5*psi**2 - a**2*V - 2*g1*X*H*psi + a**2*P
    b1 = 3*f1+6*g1*X
    b2 = -1+6*H*psi/a**2*(g1+g2*X)
    b3 = 3*f1*H**2 - 2*H*psi - a**2*V1 - 12*g2*X**2*H**2
    phi2 = invert_system(a1,a2,a3,b1,b2,b3)

    frac_aH = 1 / (a * H)
    phi_derivative = psi * frac_aH
    psi_derivative = frac_aH * phi2
    return phi_derivative, psi_derivative

def dark_energy_check(scales,Hubble_parameters,plot_title):
    # calculate dark energy fraction from 1 - radiation energy.
    dark_energy = 1 - Model.density0/(3*scales**2 * Hubble_parameters**2)
    plot_state(scales,dark_energy,"Dark Energy Omega",plot_title)

def energy_fractions(a,H,phi,psi,plot_title):
    f = Model.f(phi)
    f1 = Model.f_derivative(phi)
    g1 = Model.g_derivative(psi, a)
    rho = Model.density(a)
    X = Model.X(psi,a)
    V = Model.potential(phi)

    rho_c = 3*f*H**2/a**2
    xi_term = -3*f1*psi*H/a**2
    zeta_term = -6*g1*X*psi*H/a**2

    plt.figure()
    plt.semilogx(a,rho/rho_c,label=r"$\rho_R$")
    plt.plot(a,(X+V)/rho_c,label=r"$X+V$")
    #plt.plot(a,V/rho_c,label=r"$V$")
    plt.plot(a,zeta_term/rho_c,label=r"Cubic Galileon scale")
    plt.plot(a,xi_term/rho_c,label=r"Minimal coupling scale")
    plt.xlabel(r"$a/a_i$")
    plt.ylabel(r"$\rho/\rho_c$")
    #plt.title("Energy fractions of " + plot_title)
    plt.legend()
    plt.savefig(f"energy fractions,a={a[-1]:.1g},xi={Model.xi:.1g},zeta={Model.zeta:.1g}.pdf")
    plt.show()

def energy_analysis(k, scales, GW, GW_derivative):
    # note that the initial energy depends on k (=k**2), so we must scale it to find the damping spectrum
    final_energy = k ** 2 * GW[-1] ** 2 + GW_derivative[-1] ** 2
    return final_energy / k ** 2

def fft_frequency(a,y,print_freq=False,plot=False):
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freq = np.fft.rfftfreq(a.size,d=(a[1]-a[0]))
    if plot:
        plt.figure()
        plt.loglog(freq, spectrum)
    peak_idx = np.argmax(spectrum[1:])+1
    peak_freq = freq[peak_idx]
    if print_freq:
        print(f"Field frequency is f={peak_freq:.2g}, k is {2*np.pi*peak_freq:.2g}, period is {1 / peak_freq:.2g}")
    return peak_freq

def effective_k_slope(k_in,k_effectives):
    k_effective = np.array(k_effectives)
    mask = k_effective > 0
    k = k_in[mask]
    k_eff = k_effective[mask]
    slope = np.mean(k_eff / k)
    return slope

def alpha_B(a,psi,H):
    X = Model.X(psi,a)
    return 2*Model.zeta*X*psi/H

def background_simulator(init_state,params,scale_range,plot_title,plot_bg=True,plot_energies=False):
    # params are (in order): V0,zeta,xi,density0 (radiation density at a=1)
    # initial state is (in order): scalar field, scalar field derivative to conformal time
    set_params(params)
    # check constraint with these parameters by calculating Hubble (should be 1
    constraint_Hubble = calc_Hubble(scale_range[0],init_state[0],init_state[1])
    print(f"Checking constraint at initial state: H = {constraint_Hubble:.10g}. Should be 1.")
    t_eval = np.logspace(np.log10(scale_range[0]),np.log10(scale_range[1])-1e-8,1000)
    #t_eval = np.linspace(scale_range[0],scale_range[1],10000)

    output = solve_ivp(derivatives,scale_range,init_state,t_eval=t_eval,dense_output=True,atol=1e-9, rtol=1e-8, method='DOP853')

    scale_points, states_points = output['t'],output['y']
    field = states_points[0]
    field_derivative = states_points[1]
    Hubble = calc_Hubble(scale_points,field,field_derivative)
    print(f"Final Hubble parameter is {Hubble[-1]:.3g} at a={scale_points[-1]:.3g}")
    if plot_energies:
        energy_fractions(scale_points,Hubble,field,field_derivative,plot_title)
    field_freq = fft_frequency(scale_points,field,print_freq=False)
    k_phi = 2*np.pi*field_freq
    if plot_bg:
        state_names = ["Scalar field","Scalar field derivative","Hubble parameter"]
        log_options = [False,False,True]
        for state_name,state_points,log in zip(state_names,[field,field_derivative,Hubble],log_options):
            plot_state(scale_points,state_points,state_name,plot_title,log=log)
        dark_energy_check(scale_points,Hubble,plot_title)
    return output["sol"],k_phi,Hubble[-1],alpha_B(scale_points,field_derivative,Hubble)

def GW_derivatives(a,state,k,bg_solution):
    h,h1 = state
    phi,psi = bg_solution(a)
    H = calc_Hubble(a,phi,psi)

    f = Model.f(phi)
    f1 = Model.f_derivative(phi)
    h2 = -(2*H+f1/f*psi)*h1-k**2*h

    frac_aH = 1 / (a * H)
    h1_derivative = frac_aH * h2
    h_derivative = frac_aH * h1
    return h_derivative,h1_derivative

def GW_simulator(scale_range,k_range,bg_solution,analysis_function,plot_title,output_points=5000,plot_strain=False):
    # to calculate a damping spectrum, give a range of k
    # to calculate the strain and its derivative, give only one k in k_range
    # start with flat spectrum, so all h_ij(k),a=1)=1 and zero velocity
    h_init = 1
    h1_init = 0
    init_state = [h_init,h1_init]
    #t_eval = np.logspace(np.log10(scale_range[0]),np.log10(scale_range[1])-0.000000000001,output_points)
    t_eval = np.linspace(scale_range[0],scale_range[1],output_points)
    damping_spectrum = []
    k_effective = []

    print()
    for k in k_range:
        #print(f"k={k:.3g}     ",end="\r")
        args = k,bg_solution
        output = solve_ivp(GW_derivatives,scale_range,init_state,args=args,t_eval=t_eval,dense_output=True, atol=1e-9, rtol=1e-8,method='DOP853')
        scales, states = output['t'],output['y']
        GW = states[0]
        GW_derivative = states[1]
        k_effective.append(2*np.pi*fft_frequency(scales,GW))
        state_names = ["GW strain", "GW derivative"]
        if plot_strain:
            for state_name, state_points in zip(state_names, [GW,GW_derivative]):
                plot_state(scales, state_points, state_name, plot_title+f",k={k:2g}",k)
        if len(k_range) == 1:
            return output["sol"] # to be able to plot/analyse a single frequency
        damping = analysis_function(k,scales,GW,GW_derivative)
        damping_spectrum.append(damping)
    #print("\n")
    return k_effective,np.array(damping_spectrum)

def simulation(final_scale, analysis_function=energy_analysis, V0=0.1, zeta=1, xi=1/8,scale_xi=True,init_field = 1,dynamic_V0=False,
               DE_fraction=0.9,plot_bg=True,plot_strain=False,calc_spectrum=True,k_low=None,k_high=0,plot_energies=False,k_range=None):
    init_scale = 1
    init_field_derivative = 0
    # we can allow the initial field to be non-zero, as long as xi*phi**2<<1
    # if xi is None, as per default, it is automatically scaled from 1/8
    if scale_xi:
        xi /= init_field**2
        print(f"Dynamically scaled xi={xi:.2g} to match the initial field={init_field}")
    if xi*init_field**2 >= 1:
        raise ValueError(f"xi*phi**2 is bigger than 1.")
    # set V0 such that the DE fraction is initially 0.9
    if dynamic_V0:
        V0 = 3*(1-xi*init_field**2)*DE_fraction/init_field**4
        print(f"Dynamically calculated V0={V0:2g} with DE fraction={DE_fraction}.")
    # This is the radiation energy density. Determined by flat universe condition
    density0 = 3 * (1 - xi*init_field**2) - V0*init_field**4
    print(f"Initial density is {density0:3g}")
    if density0 < 0:
        raise ValueError(f"negative radiation density.")
    params = [V0, zeta, xi, density0]
    scale_range = [init_scale, final_scale]
    init_state = [init_field, init_field_derivative]

    plot_title = rf"$\rho_i$={density0:.2g},$\xi$={xi:.2g},$\zeta$={zeta:.2g},$V_0$={V0:.2g},fDE={DE_fraction},phi_i={init_field},a={final_scale}"
    bg_solution,k_phi,H_final,alpha_B_array = background_simulator(init_state, params, scale_range, plot_title,plot_bg,plot_energies)
    if calc_spectrum:
        if k_range is None:
            k_amount = 200 if not plot_strain else 3
            if k_low is None:
                k_low = -np.log10(final_scale) if not plot_strain else -1
            k_range = np.logspace(k_low, k_high, k_amount)
        k_effective,GW_spectrum = GW_simulator(scale_range, k_range, bg_solution, analysis_function, plot_title,plot_strain=plot_strain)
        return k_range,k_effective,GW_spectrum,k_phi,H_final,alpha_B_array
    else:
        return bg_solution,k_phi,H_final,alpha_B_array

def main():
    def dummy_analysis_function(k,scales,GW,GW_derivative):
        return None
    simulation(final_scale=100,analysis_function=dummy_analysis_function,plot_strain=True,init_field=2)

if __name__ == "__main__":
    main()