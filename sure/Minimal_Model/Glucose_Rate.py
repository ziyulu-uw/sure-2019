# Author: Xinyu Li
# Email: xl1796@nyu.edu
# Date: July 2019
# Description: This program contains Glucose rate Ra input given in Paper 1

from Parameters import glucose
def Ra(t):
    """It is a Dirac Delta approximation function (integral sum may not to 1)
    Glucose Rate Ra input in paper 1 Fig.3"""

    #peak = glucose.R       # unit: mmol/min
    peak = 15
    lasting_time = 1  #unit: min

    return (t<= lasting_time) * peak
