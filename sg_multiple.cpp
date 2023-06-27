#include <scipy,time>
#include <numpy>
#include <pandas>
#include <numpy.random>

#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

std::vector<double> equations(double t, std::vector<double> y, int NumSpecies, std::vector<double> growth_rate, std::vector<double> Sigma, std::vector<std::vector<double>> interaction, double dilution)
{
    std::vector<double> eq; 
    for (int i = 0; i < NumSpecies; i++)
    {
        std::vector<double> temp;
        for (int j = 0; j < NumSpecies; j++)
        {
            temp.push_back(interaction[j][i] * y[i]);
        }
        double PosSum = 0;
        double NegSum = 0;
        for (int k = 0; k < temp.size(); k++)
        {
            if (temp[k] > 0)
            {
                PosSum += temp[k];
            else
            {
                NegSum += temp[k];
            }
        }
        eq.push_back(growth_rate[i] * y[i] * (1 + NegSum + Sigma[i] * PosSum / (1 + PosSum)) - dilution * y[i]);
    }
    return eq;
}


std::vector<double> GLV(std::vector<double> timespan, std::vector<double> initial, int NumSpecies, std::vector<double> growth_rate, std::vector<double> Sigma, std::vector<std::vector<double>> interaction, double dilution)
{
    std::vector<double> sol = scipy.integrate.solve_ivp(equations,
                                                       t_span = timespan,
                                                       y0 = initial,
                                                       method = 'RK45',
                                                       args=(NumSpecies, growth_rate, Sigma, interaction, dilution),
                                                       dense_output=True)
    return solve_ivp


}

vector<double> main(double argc, char **argv)
{
    
}

std::

