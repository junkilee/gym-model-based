import os, sys
import numpy as np

if len(sys.argv) != 4:
    print("python output.py EPSILON START_ID END_ID!")
    sys.exit(1)

test_epsilon = sys.argv[1]
start_id = int(sys.argv[2])
max_id = int(sys.argv[3])

model = []
real = []

for i in range(start_id, max_id + 1):
    output_file = "outputs/output_dqn_" + test_epsilon + "_" + str(i) + ".txt"
    f = open(output_file, "r")
    first = map(float, f.readline().split(","))
    model += first
    second = map(float, f.readline().split(","))
    real += second

print("{} +- {}\n".format(np.mean(model), np.std(model) * 1.97 / np.sqrt(len(model))))
print("{} +- {}\n".format(np.mean(real), np.std(real) * 1.97 / np.sqrt(len(real))))
