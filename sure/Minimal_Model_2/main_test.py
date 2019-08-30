from Path_whole import generate_path_whole
from Initialization import init_cond, N, N_meas, param_list, Gb, Ib, T, T_list, meal_params,total_t_list
from Noise_generator import noise_path
import matplotlib.pyplot as plt

Filter = [0.1, 0.001, 0.1, 0.03]
control_gain = [15, 3, 0.1, 0.5]
total_noise =  noise_path(init_cond, N*N_meas)
model_state_variable, Z, true_state_variable = generate_path_whole(init_cond, param_list, control_gain, Filter, total_noise, Gb, Ib, N_meas, T, T_list, N, meal_params)

## Plot G
plt.plot(total_t_list, model_state_variable[0], label = "model estimation")
plt.plot(total_t_list, true_state_variable[0], label = "in-silico subject")
plt.ylabel("G")
plt.xlabel("t")
plt.title("Glucose $G(t)$")
plt.legend()
plt.show()


## Plot Ra
plt.plot(total_t_list, model_state_variable[3], label = "model estimation")
plt.plot(total_t_list, true_state_variable[3], label = "in-silico subject")
plt.ylabel("Ra")
plt.xlabel("t")
plt.title("Glucose $Ra(t)$")
plt.legend()
plt.show()