# Traffic-Prediction-with-T-GCN-and-POI-Data
The code for project "Traffic Prediction based on Spatio-Temporal Modeling and Alternative Data Mining" in CS173, ShanghaiTech University.

## Introduction
Accurate and real-time traffic prediction plays an important role in intelligent transportation systems and is of great significance for urban traffic planning, traffic management and traffic control.We propose a new traffic prediction method, expecting to fuse spatio-temporal data and a series of knowledge-driven data to build an innovative network structure, and combine it with the POI (i.e., point of interest) data to try to predict urban traffic flow, expecting to achieve better results.

## Code Structure
```
|--Traffic-Prediction-with-T-GCN-and-POI-Data
|   |--MapReduce             /* Folder for MapReduce
|      |--PreProcess.py      /* Pre-process for MapReduce
|      |--Mapper.py          /* Map function
|      |--Reducer.py         /* Reduce function
|      |--sz_adj.csv         /* Input file for MapReduce section
|      |--temp_data.txt      /* Input file for Map function
|      |--sz_edge.csv        /* Output file for MapReduce section
|   |--data             /* Folder for data
|      |--... (csv files)
|   |--libs             /* Folder for libs
|      |--...
|   |--logs             /* Folder for saved model and other running logs
|      |--...
|   |--results          /* Folder for evaluation results
|      |--...
|   |--dataloader.py                /* Files for loading data
|   |--main.py                      /* Entry of the program
|   |--models.py                    /* Files for building model
|   |--utils.py                     /* Model training & Testing
|   |--qtraffic_visualizaiton.py                     /* Q-Traffic visualization
```

### Running Pipeline
For example, to run the program with gru on sz_taxi:

First, you need to download the data from [here](https://github.com/lehaifeng/T-GCN/tree/master/AST-GCN/data).

Make sure your data follows the name convention of `${DATA_PREFIX}_${DATATYPE}.csv`, and put them under the folder `data`.

Then, you need to run the following command:

```
$ python main.py --model gru --data_name sz --log_dir ./logs/sz
```
Other running configurations are listed in the main.py. And you can run with other data with these similar steps.

If you want to run the program with Q-traffic dataset, please download the data from [here](https://github.com/JingqingZ/BaiduTraffic).

### MapReduce
MapReduce is a parallel programming model and methodology. We use it to process our data to improve the efficiency of our method.

#### Usage
We deploy the Hadoop on Windows11 with JDK8, Hadoop-3.1.3 and apache-hadoop 3.1.0 winutils. We use the hadoop-streaming-3.1.3.jar to run the MapReduce code. There are two ways to run the MapReduce code.

To run it locally,
```
cd ./MapReduce
python PreProcess.py
cat temp_data.txt | python Mapper.py | python Reducer.py
```

To run it on the Hadoop distributed computing platform,

first you need to install Java SE Development Kit 8, and download Hadoop-3.1.3 and apache-hadoop 3.1.0 winutils. Then you need to replace the bin folder under Hadoop-3.1.3 with the bin folder under apach-hadoop 3.1.0 winutils. After that, you need to set the environment variable of JAVA_HOME and HADOOP_HOME to where you install them. After you set them correctly, you can start run the MapReduce code.

First activate hadoop by double click the start-all.cmd under "hadoop-3.1.3/sbin/start-all.cmd". Then you need to upload your file to the Hadoop and run:
```
hdfs dfs -mkdir /user/test
cd ./MapReduce
hdfs dfs -put ./temp_data.txt /user/test
hdfs dfs -put ./Mapper.py /user/test
hdfs dfs -put ./Reducer.py /user/test
cd hadoop-3.1.3/share/hadoop/tools/lib/
hdfs jar hadoop-streaming-3.1.3.jar -file Mapper.py -mapper Mapper.py -file Reducer.py -reducer Reducer.py -input /user/test -output /user/test/output
```

## Conclusion

We design and implement a traffic predicting system that combine spatio-temporal based multidimensional data network and combine it with alternative data like POI and weather. The experiments on SZ-Taxi and Q-Traffics datasets show that our model can achieve the highest accuracy compared with GRU and the SOTA traffic prediction method(i.e., A3T-GCN). We have also conducted analysis and experiments for combining alternative data with GRU and A3T-GCN, as a result, the methods combining alternative data were all better performing than the original methods. Limited by time and computing resources, we cannot further explore how to embed more alternative data into our traffic predicting pipeline to achieve better performance, but our results prove that the combination of alternative data and the traditional traffic predicting method is feasible and effective.
