import os, sys
import numpy as np
import matplotlib.pyplot as plt

def retrieve_data(epsilon, s_id, e_id):
    real = []

    for i in range(s_id, e_id + 1):
        output_file = "outputs/output_off_" + str(epsilon) + "_" + str(i) + ".txt"
        try:
            f = open(output_file, "r")
            second = map(float, f.readline().split(","))
            real += [np.mean(second)]
        except:
            pass

    return real

if len(sys.argv) != 3:
    print("python graph_off.py START_ID END_ID!")
    sys.exit(1)

epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])

reals_mean = []
reals_int = []

for ep in epsilons:
    real = retrieve_data(ep, start_id, end_id)
    real_sqrt_n = np.sqrt(len(real))
    print("epsilon = {}".format(ep))
    print("{} +- {}\n".format(np.mean(real), np.std(real) * 1.97 / real_sqrt_n))
    reals_mean += [np.mean(real)]
    reals_int += [np.std(real) * 1.97 / real_sqrt_n]

plt.figure()
plt.errorbar(epsilons, reals_mean, reals_int, label="real")
plt.title('avg of avg')
plt.xlabel('epsilon')
plt.ylabel('avg. rew.')
plt.legend()
plt.savefig('graphs/1_off_avgavg_result.pdf')
