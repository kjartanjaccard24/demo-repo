'''comparison case against vac with no flux soak and rectangular, prescribed trajectory'''
from traceback import format_exc
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('ISM-Plasma-main')

from ismplasma.utils.constants import K_to_eV
from ismplasma.utils.logger import Logger
from ismplasma.utils import plot_tools
from ismplasma.gs.mesh import UniformRadialSkewMesh
from ismplasma.profile import CenteredPiecewiseLinear
from ismplasma.plasma import TwoTempPlasma

plot_tools.disable_warnings()


def _main():
    nr, nz, nc = 50, 60, 15
    dt = 10e-6
    t_start = 0.
    t_end = 10e-3

    # trajectory
    t_comp = 3e-3
    H = 1.6
    z = np.linspace(-H/2, H/2, nz+1)
    r_inner = 0.06+0.40*(2*z/H)**2
    r_inner = 0.1

    def r_outer_fun(t):
        r_outer = (1.1- 0*t/t_comp*0.575) + 0*t/t_comp*(-0.1*(2*z/H)**2 + (0.25+0.1)*(2*z/H)**4)
        return r_outer
    r_outer = r_outer_fun(t_start)
    mesh = UniformRadialSkewMesh(r_inner, r_outer, z, nr)

    # plasma profiles/parameters
    psi_n_1d = np.linspace(0, 1, nc+1)
    F_lcs = 0.5
    F_loss = 0.0
    f_const = 0.0885
    F = CenteredPiecewiseLinear(psi_n_1d, fun=lambda x: F_lcs + f_const*x**2)
    n_e = CenteredPiecewiseLinear(psi_n_1d, fun=lambda x: np.full_like(x, 1e20))
    #T_e = CenteredPiecewiseLinear(psi_n_1d, fun=lambda x: 300/K_to_eV*x)
    #T_e = CenteredPiecewiseLinear(psi_n_1d, fun=lambda x: np.full_like(x,300/K_to_eV))

    # sigmoid temperature profile
    T_e = CenteredPiecewiseLinear(psi_n_1d, fun=lambda x: (300/K_to_eV)*((1-0.1)*(1-((x-1)**2))**2 + 0.1))    
    T_i = T_e * 1.5
    print(T_e.x)
    print(T_e.data*K_to_eV)
    print(F.x)
    print(F.data)
    #exit()

    # psi boundary condition
    psi_outer = -0.0
    bc = np.zeros_like(mesh.x)
    bc[-1, :] = psi_outer  # outer boundary
    bc[:, 0] = bc[:, -1] = np.linspace(0, psi_outer, bc.shape[0])  # top and bottom

    plasma = TwoTempPlasma(mesh, F, F_lcs, n_e, T_e, T_i, bc=bc,
                           eta='spitzer', chi_e=3., chi_i=3., Z_avg=1.2, Z_eff=1.5,
                           q_solver_options={'atol': 1e-6, 'maxiter': 10000,
                                             'alpha': 0.5, 'alpha_rhs': 0.1,
                                             'raise_on_fail': True})

    logger = Logger()
    logger.add_log(plasma, 'n_e', fig='Density', z=plasma.F.data_x, cbar_label=r'$\psi_n$')
    logger.add_log(plasma, 'T_e', fig='Temperature', ls='--', z=plasma.F.data_x, cbar_label=r'$\psi_n$')
    logger.add_log(plasma, 'T_i', fig='Temperature', z=plasma.F.data_x)
    logger.add_log(plasma, 'F', fig='F', z=plasma.F.data_x, cbar_label=r'$\psi_n$')
    logger.add_log(plasma, 'F_lcs', fig='F', ls='--', c='k')
    logger.add_log(plasma, 'F_prime', fig='lambda', z=plasma.F.x, cbar_label=r'$\psi_n$')
    logger.add_log(plasma, 'centroid', fig='lambda centroid')
    logger.add_log(plasma, 'q', fig='q', z=plasma.F.data_x, cbar_label=r'$\psi_n$')
    logger.add_log(plasma, 'q0', fig='q', c='k', ls='--')
    logger.add_log(plasma, 'q_edge', fig='q', c='k')
    logger.add_log(plasma, 'beta_t', fig='beta')
    logger.add_log(plasma, 'beta_t_troyon', fig='beta')
    logger.add_log(plasma, 'beta_t_freidberg', fig='beta')
    logger.add_log(plasma, 'beta_t_bernard', fig='beta')
    logger.add_log(plasma, 'beta_p', fig='beta')
    logger.add_log(plasma, 'greenwald_ratio', fig='Greenwald')
    logger.add_log(plasma, 'a', fig='Radii')
    logger.add_log(plasma, 'R0', fig='Radii')
    logger.add_log(plasma, 'kappa', fig='Geometry')
    logger.add_log(plasma, 'epsilon', fig='Geometry')
    logger.add_log(plasma, 'trian', fig='Geometry')
    logger.add_log(plasma, 'k', fig='k')
    logger.add_log(plasma, 'psi_1d', fig='psi_1d', z=plasma.psi_1d.data_x, cbar_label=r'$\psi_n$')
    logger.add_log(plasma, 'Phi_enc', fig='Flux')
    logger.add_log(plasma, 'Psi_enc', fig='Flux')
    logger.add_log(plasma, 'I_plasma', fig='Current')
    logger.add_log(plasma, 'I_pol', fig='Current')
    logger.add_log(plasma, 'I_shaft', fig='Current')
    logger.add_log(plasma, 'E_pol_lcs', fig='Energy')
    logger.add_log(plasma, 'E_tor', fig='Energy')
    logger.add_log(plasma, 'E_tor_vac', fig='Energy')
    logger.add_log(plasma, 'E_tor_diff', fig='Energy')
    logger.add_log(plasma, 'E_th', fig='Energy')
    logger.add_log(plasma, 'l_i1', fig='Internal inductance')
    logger.add_log(plasma, 'dd_he3_fusion_rate', fig='Neutron Rate', slc=np.s_[:],
                   logy=True, z=plasma.F.data_x, cbar_label=r'$\psi_n$')

    logger.add_log(plasma, 'p', fig='pressure')
    logger.record_all(0)

    plt.figure('rj, t=0')
    plasma.q_solution.plot_rj()

    plt.figure('psi, t=0')
    plasma.q_solution.plot_psi()
    plt.figure('contours, t=0')
    plasma.q_solution._plot_contour_set()
    plt.figure('pressure, t=0')
    plasma.p.plot()
    plt.figure('F, t=0')
    plasma.F.plot()

    plt.figure('q profile')
    plasma.q.plot()
    plt.figure('n_e profile')
    plasma.n_e.plot()
    plt.figure('lambda profile')
    plasma.F_prime.plot()
    plt.figure('T profiles')
    plasma.T_e.plot(ls='--', label='T_e')
    plasma.T_i.plot(c=plot_tools.get_last_color(), label='T_i')
    plt.legend()
    plt.figure('eV T profiles')
    plt.plot(plasma.T_i.data_x,plasma.T_i.data*K_to_eV)
    plt.plot(plasma.T_e.data_x,plasma.T_e.data*K_to_eV)
    plt.grid(True)



    L0 = mesh.boundary_contour.L  # inductance of entire domain
    t = 0.
    timer = perf_counter()
    try:
        for t in np.arange(t_start+dt, t_end+dt/2, step=dt):
            print(t)
            r_outer = r_outer_fun(t)
            mesh = UniformRadialSkewMesh(r_inner, r_outer, z, nr)
            L = mesh.boundary_contour.L
            F_lcs_t = F_lcs * L0/L*(1 - F_loss*t/t_end)
            plasma.calculate_updates(dt, mesh=mesh, F_lcs=F_lcs_t, rk2=True)
            plasma.update(t)
    except Exception:  # pylint: disable=broad-exception-caught
        print(f'Exception during time loop: {format_exc()}')
    print(f'Elapsed wall-clock time: {perf_counter()-timer} s')

    plt.figure(f'rj, t={t}')
    plasma.q_solution.plot_rj()

    plt.figure(f'psi, t={t}')
    plasma.q_solution.plot_psi()

    plt.figure('q profile')
    plasma.q.plot()
    print(plasma.q.data_x)
    print(plasma.q.data)

    plt.figure('n_e profile')
    plasma.n_e.plot()
    plt.figure('lambda profile')
    plasma.F_prime.plot()
    plt.figure('T profiles')
    plasma.T_e.plot(ls='--')
    plasma.T_i.plot(c=plot_tools.get_last_color())
    logger.plot_all()
    plt.show()


if __name__ == '__main__':
    _main()
