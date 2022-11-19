import sys

count = 0
for line in sys.stdin:
    word_list = line.strip().split(",")
    for i in range(len(word_list)):
        road = word_list[i]
        if road == "1":
            print("%s,%s" % (count, i-1))
    count += 1

    # for word in word_list:
    #     print(word+"\t1")
    # id = word_list[1]
    # speedStr = word_list[3]
    # speed = float(speedStr)
    # if speed < 15:
    #     print(id)
