import os, sys
import numpy as np
import matplotlib.pyplot as plt

def retrieve_data(epsilon, s_id, e_id):
    model = []
    model1 = []
    real = []

    for i in range(s_id, e_id + 1):
        output_file = "outputs/output_dqn_" + str(epsilon) + "_" + str(i) + ".txt"
        f = open(output_file, "r")
        first = map(float, f.readline().split(","))
        model += [np.mean(first)]
        middle = map(float, f.readline().split(","))
        model1 += [np.mean(middle)]
        #model += first
        second = map(float, f.readline().split(","))
        real += [np.mean(second)]
        #real += second

    return model, model1, real

if len(sys.argv) != 3:
    print("python graph.py START_ID END_ID!")
    sys.exit(1)

epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
#epsilons = [0.2, 0.4, 0.6, 0.8, 1.0]
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])

models_mean = []
models_int = []
model1_mean = []
model1_int = []
reals_mean = []
reals_int = []

for ep in epsilons:
    model, model1, real = retrieve_data(ep, start_id, end_id)
    model_sqrt_n = np.sqrt(len(model))
    model1_sqrt_n = np.sqrt(len(model1))
    real_sqrt_n = np.sqrt(len(real))
    print("epsilon = {}".format(ep))
    print("{} +- {}\n".format(np.mean(model), np.std(model) * 1.97 / model_sqrt_n))
    print("{} +- {}\n".format(np.mean(model1), np.std(model1) * 1.97 / model1_sqrt_n))
    print("{} +- {}\n".format(np.mean(real), np.std(real) * 1.97 / real_sqrt_n))
    models_mean += [np.mean(model)]
    models_int += [np.std(model) * 1.97 / model_sqrt_n]
    model1_mean += [np.mean(model1)]
    model1_int += [np.std(model1) * 1.97 / model1_sqrt_n]
    reals_mean += [np.mean(real)]
    reals_int += [np.std(real) * 1.97 / real_sqrt_n]

plt.figure()
plt.errorbar(epsilons, models_mean, models_int, label="model")
plt.errorbar(epsilons, model1_mean, model1_int, label="model 0.0")
plt.errorbar(epsilons, reals_mean, reals_int, label="real")
plt.title('avg of avg')
plt.xlabel('epsilon')
plt.ylabel('avg. rew.')
plt.legend()
plt.savefig('graphs/7__avgavg_result.pdf')
#plt.figure()
#plt.errorbar(epsilons, models_mean, models_int)
#plt.savefig('graphs/2_result_model.pdf')
#plt.figure()
#plt.errorbar(epsilons, reals_mean, reals_int)
#plt.savefig('graphs/2_result_real.pdf')
