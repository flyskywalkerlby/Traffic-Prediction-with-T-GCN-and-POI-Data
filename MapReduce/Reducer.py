import sys
import numpy as np

current_word = None
sum = 0
roadDict = {}
roadArr = [[], []]

for line in sys.stdin:
    out = line.strip()
    out = out.split(",")
    roadArr[0].append(int(out[0]))
    roadArr[1].append(int(out[1]))
    print("%s,%s" % (out[0], out[1]))
# print(roadArr[0])
# print(roadArr[1])
npArr = np.array(roadArr)
np.savetxt('sz_edges.csv', npArr, delimiter=',')
