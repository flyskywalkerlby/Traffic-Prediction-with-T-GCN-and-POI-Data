import numpy as np
import pandas as pd
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric.utils import dense_to_sparse
import os
import torch


class TrafficDataloader(METRLADatasetLoader):

    def __init__(self, raw_data_dir, args):
        super(METRLADatasetLoader, self).__init__()

        self.args = args
        self.raw_data_dir = raw_data_dir
        self._read_data()
        np.save(os.path.join(self.args.log_dir, 'mean.npy'), self.X_mean)
        np.save(os.path.join(self.args.log_dir, 'std.npy'), self.X_std)

    def _read_data(self):
        X, p, w, A = self.load_data(self.raw_data_dir, self.args.data_name, self.args.use_poi, self.args.use_weather)
        if len(X.shape) == 2:  # add a feature dim if it only has one feature
            X = np.expand_dims(X, axis=1)

        target_X = [X]
        if self.args.use_poi:
            if len(p.shape) == 2:
                p = np.expand_dims(p, axis=1)
            target_X.append(p)
        if self.args.use_weather:
            if len(w.shape) == 2:
                w = np.expand_dims(w, axis=1)
            target_X.append(w)

        X = np.concatenate(target_X, axis=1)
        self.X_mean = np.mean(X, axis=(0, 2)).reshape(1, -1, 1)
        self.X_std = np.std(X, axis=(0, 2)).reshape(1, -1, 1)
        X = (X - self.X_mean) / self.X_std

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):

        if self.args.use_mapreduce:
            self.edges = self.A
            self.edge_weights = None
        else:
            edge_indices, values = dense_to_sparse(self.A)
            edge_indices = edge_indices.numpy()
            values = values.numpy()
            self.edges = edge_indices
            self.edge_weights = values

    @staticmethod
    def check_include_header(file_path):
        checker = pd.read_csv(file_path, nrows=1, header=None)
        # check is the first 8 elements are increased by 1
        for i in range(8):
            if checker.iloc[0, i] != checker.iloc[0, i+1] - 1:
                return False
        return True

    def load_data(self, root_dir: str, prefix, use_poi, use_weather):
        tf_path = os.path.join(root_dir, f'{prefix}_speed.csv')
        tf = pd.read_csv(tf_path, header=0 if self.check_include_header(tf_path) else None)
        tf = np.mat(tf, dtype=np.float32) if 'bj' in prefix else np.mat(tf, dtype=np.float32).T
        poi, weather = None, None

        if use_poi:
            poi_path = os.path.join(root_dir, f'{prefix}_poi.csv')
            poi = pd.read_csv(poi_path, header=0 if self.check_include_header(poi_path) else None)
            poi = np.mat(poi, dtype=np.float32)
            poi = poi[:, 0].repeat(tf.shape[1], axis=1) if 'bj' in prefix else poi
        if use_weather:
            weather_path = os.path.join(root_dir, f'{prefix}_weather.csv')
            weather = pd.read_csv(weather_path, header=0 if self.check_include_header(weather_path) else None)
            weather = np.mat(weather, dtype=np.float32).T

        if self.args.use_mapreduce:
            adj_path = os.path.join(root_dir, f'{prefix}_edges.csv')
        else:
            adj_path = os.path.join(root_dir, f'{prefix}_adj.csv')
        adj = pd.read_csv(adj_path, header=0 if self.check_include_header(adj_path) else None)
        adj = np.mat(adj, dtype=np.int64)

        return tf, poi, weather, adj

    def _read_web_data(self):
        raise NotImplementedError("web version no usable")


if __name__ == '__main__':
    tmp = TrafficDataloader()
    print(tmp.X.shape)
