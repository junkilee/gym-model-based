import os, sys
import numpy as np
import matplotlib.pyplot as plt

def retrieve_data(portion, epsilon, s_id, e_id):
    model = []
    model1 = []
    real = []

    for i in range(s_id, e_id + 1):
        output_file = "outputs/output_dqn_" + str(portion) + "_" + str(epsilon) + "_" + str(i) + ".txt"
        try:
            f = open(output_file, "r")
            first = map(float, f.readline().split(","))
            model += [np.mean(first)]
            middle = map(float, f.readline().split(","))
            model1 += [np.mean(middle)]
            #model += first
            second = map(float, f.readline().split(","))
            real += [np.mean(second)]
            #real += second
        except:
            pass

    return model, model1, real

def retrieve_data_off(portion, s_id, e_id):
    real = []

    for i in range(s_id, e_id + 1):
        output_file = "outputs/output_off_" + str(portion) + "_" + str(i) + ".txt"
        try:
            f = open(output_file, "r")
            second = map(float, f.readline().split(","))
            real += [np.mean(second)]
        except:
            pass

    return real
if len(sys.argv) != 4:
    print("python graph.py OUTPUT_ID START_ID END_ID!")
    sys.exit(1)

#epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
portions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
epsilon = 0.0
#epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
out_id = sys.argv[1]
start_id = int(sys.argv[2])
end_id = int(sys.argv[3])

models_mean = []
models_int = []
model1_mean = []
model1_int = []
reals_mean = []
reals_int = []
offs_mean = []
offs_int = []
box_real_mean = []
box_off_mean = []

for por in portions:
    model, model1, real = retrieve_data(por, epsilon, start_id, end_id)
    off = retrieve_data_off(por, start_id, end_id)
    model_sqrt_n = np.sqrt(len(model))
    model1_sqrt_n = np.sqrt(len(model1))
    real_sqrt_n = np.sqrt(len(real))
    off_sqrt_n = np.sqrt(len(off))
    print("portion = {}".format(por))
    print("epsilon = {}".format(epsilon))
    print("{} +- {}\n".format(np.mean(model), np.std(model) * 1.97 / model_sqrt_n))
    print("{} +- {}\n".format(np.mean(model1), np.std(model1) * 1.97 / model1_sqrt_n))
    print("{} +- {}\n".format(np.mean(real), np.std(real) * 1.97 / real_sqrt_n))
    print("{} +- {}\n".format(np.mean(off), np.std(off) * 1.97 / off_sqrt_n))
    models_mean += [np.mean(model)]
    models_int += [np.std(model) * 1.97 / model_sqrt_n]
    model1_mean += [np.mean(model1)]
    model1_int += [np.std(model1) * 1.97 / model1_sqrt_n]
    reals_mean += [np.mean(real)]
    box_real_mean += [real]
    box_off_mean += [off]
    reals_int += [np.std(real) * 1.97 / real_sqrt_n]
    offs_mean += [np.mean(off)]
    offs_int += [np.std(off) * 1.97 / off_sqrt_n]
    

plt.figure()
plt.errorbar([1,2,3,4,5,6,7,8,9,10], reals_mean, reals_int, label="real", fmt='g')
plt.boxplot(box_real_mean, labels=portions)
plt.ylim((-50, 1050))
plt.xlabel('data sparsity')
plt.ylabel('avg. rew.')
plt.title('Agents learning directly from the learned model')
plt.savefig('graphs/' + out_id + '_box_real_result.pdf')

plt.figure()
plt.errorbar([1,2,3,4,5,6,7,8,9,10], offs_mean, offs_int, label="offline", fmt='r')
plt.boxplot(box_off_mean, labels=portions)
plt.ylim((-50, 1050))
plt.xlabel('data sparsity')
plt.ylabel('avg. rew.')
plt.title('Offline batch Q learning')
plt.savefig('graphs/' + out_id + '_box_off_result.pdf')

#plt.figure()
#plt.errorbar(portions, models_mean, models_int, label="model")
#plt.errorbar(portions, reals_mean, reals_int, label="real")
#plt.errorbar(portions, offs_mean, offs_int, label="offline")
#plt.title('avg of avg')
#plt.xlabel('data sparsity')
#plt.ylabel('avg. rew.')
#plt.legend(loc=4)
#plt.savefig('graphs/' + out_id + '_portion_avgavg_result.pdf')
