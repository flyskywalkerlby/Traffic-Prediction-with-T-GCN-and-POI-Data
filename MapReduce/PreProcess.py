# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv('sz_adj.csv', encoding='utf-8')
    with open('temp_data.txt', 'w', encoding='utf-8') as f:
        print(len(data))
        for line in data.values:
            for item in line:
                f.write((str(item) + ','))
            f.write('\n')
        f.close()

