#include "glv.h"

std::vector<double> GLV(std::vector<double> timespan, std::vector<double> initial, int NumSpecies, std::vector<double> growth_rate, std::vector<double> Sigma, std::vector<std::vector<double>> interaction, double dilution)
{
    std::vector<double> sol = scipy.integrate.solve_ivp(equations,
                                                       t_span = timespan,
                                                       y0 = initial,
                                                       method = 'RK45',
                                                       args=(NumSpecies, growth_rate, Sigma, interaction, dilution),
                                                       dense_output=True)
    return solve_ivp;
}
