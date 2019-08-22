# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: August 2019
# Description: Meal intake function are defined here


def D(t, tk_list, qk_list, meal_time):
    """meal intake function is a piecewise constant one, starting from meal time, and last for 5mins
    @:param tk_list:     a list of time when meal intake happens
    @:param qk_listL     a list of meal intake value"""

    for i in range(len(tk_list)):
        if t-tk_list[i]<meal_time and t>tk_list[i]:
            return qk_list[i]
    return 0