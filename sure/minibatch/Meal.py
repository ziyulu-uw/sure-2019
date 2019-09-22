# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Meal intake function are defined here


def D(t, sim_idx, T, meal_params):
    """meal intake function is a piecewise constant one, starting from meal time, and last for 5mins
    @:param t:           a scalar about time
    @:param sim_idx:     an index indicates which simulation it is
    @:param T:           time period for a control
    @:param meal_params: (tk_list, qk_list, meal_time)
                          tk_list:     a list of time when meal intake happens
                          qk_list:     a list of meal intake value"""

    tk_list, qk_list, meal_time = meal_params
    t = sim_idx*T + t       # convert t in a control simulation to the corresponding time in the whole simulation
    for i in range(len(tk_list)):
        if t-tk_list[i]<meal_time and t>tk_list[i]:
            return qk_list[i]
    return 0

