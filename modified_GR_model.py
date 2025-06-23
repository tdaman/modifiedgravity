V0 = 1
xi=1
zeta = 1
density0 = 1

def X(psi, scale):
        return psi**2/(2*scale**2)

def potential(phi):
    return V0*phi**4

def potential_derivative(phi):
    return 4*V0*phi**3

def g(psi,scale):
    return zeta*X(psi,scale)

def g_derivative(psi,scale):
    return zeta

def g_derivative2(psi,scale):
    return 0

def f(phi):
    return 1-xi*phi**2

def f_derivative(phi):
    return -2*xi*phi

def f_derivative2(phi):
    return -2*xi

def density(scale):
    return density0*scale**-4

def pressure(scale,w=1/3):
    # using the equation of state for radiation
    return w*density(scale)


