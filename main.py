import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import Trainer, Tester
from dataloader import TrafficDataloader
import numpy as np
import os
from glob import glob
from torch_geometric_temporal.signal import temporal_signal_split
from models import *


def parse():
    parser = argparse.ArgumentParser()

    # Model ------------------------------------------------------------------------------------------------------------
    parser.add_argument('--model', default="a3t", type=str, help="'a3t' or 'ast'")

    # Configs ----------------------------------------------------------------------------------------------------------
    parser.add_argument('--device', default="cuda:0", type=str, help="running device")
    parser.add_argument('--log_dir', default="./logs", type=str, help='logs directory')

    # Data arguments ---------------------------------------------------------------------------------------------------
    parser.add_argument('--train_split_ratio', default=0.8, type=float, help="ratio of training set")
    parser.add_argument('--data_name', default='sz', type=str, help='dataset name')
    parser.add_argument('--time_len', default=12, type=int, help='input time sequence length')
    parser.add_argument('--pre_time_len', default=3, type=int, help='prediction time sequence length')
    parser.add_argument('--use_poi', default=1, type=int, help='whether to use poi')
    parser.add_argument('--use_weather', default=1, type=int, help='whether to use weather')
    parser.add_argument('--use_mapreduce', default=0, type=int, help='whether to use mapreduce to get edges info')

    # Training arguments -----------------------------------------------------------------------------------------------
    parser.add_argument('--epochs', default=376, type=int, help='max number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--noise_std', default=0.1, type=float, help='the standard deviation of Gaussian noise')

    # Testing arguments ------------------------------------------------------------------------------------------------
    parser.add_argument('--test_batch_size', default=1, type=int, help='test batch size')

    return parser.parse_args()


def main(args):
    if args.model == "a3t":
        TemporalGNN = TemporalGNN_a3t
    elif args.model == "a3t_vanilla":
        TemporalGNN = TemporalGNN_a3t_vanilla
    elif args.model == "gru":
        TemporalGNN = GRU
    else:
        raise ValueError("Invalid model name")

    DEVICE = torch.device(args.device)
    args.log_dir = os.path.join(
        args.log_dir, f"{args.model}_{args.data_name}_poi:{args.use_poi}_weather:{args.use_weather}"
    )
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    with open(f'./{args.log_dir}/args.log', 'w') as f:
        argsDict = args.__dict__
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
    assert args.test_batch_size == 1, "test batch size must be 1"

    loader = TrafficDataloader(raw_data_dir=os.path.join(os.getcwd(), "data"), args=args)
    dataset = loader.get_dataset(num_timesteps_in=args.time_len, num_timesteps_out=args.pre_time_len)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_split_ratio)
    train_dataset, val_dataset = temporal_signal_split(train_dataset, train_ratio=7 / 8)
    edge_index = train_dataset[0].edge_index

    train_input, val_input, test_input = (
        np.array(train_dataset.features), np.array(val_dataset.features), np.array(test_dataset.features)
    )
    train_target, val_target, test_target = (
        np.array(train_dataset.targets), np.array(val_dataset.targets), np.array(test_dataset.targets)
    )
    train_x_tensor, val_x_tensor, test_x_tensor = (
        torch.from_numpy(train_input).type(torch.FloatTensor),
        torch.from_numpy(val_input).type(torch.FloatTensor),
        torch.from_numpy(test_input).type(torch.FloatTensor)  # (B, N, F, T)
    )
    train_target_tensor, val_target_tensor, test_target_tensor = (
        torch.from_numpy(train_target).type(torch.FloatTensor),
        torch.from_numpy(val_target).type(torch.FloatTensor),
        torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)
    )
    train_dataset_new, val_dataset_new, test_dataset_new = (
        TensorDataset(train_x_tensor, train_target_tensor),
        TensorDataset(val_x_tensor, val_target_tensor),
        TensorDataset(test_x_tensor, test_target_tensor)
    )
    train_loader, val_loader, test_loader = (
        DataLoader(train_dataset_new, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4),
        DataLoader(val_dataset_new, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4),
        DataLoader(test_dataset_new, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=2)
    )

    # Training ---------------------------------------------------------------------------------------------------------
    model = TemporalGNN(
        node_features=train_x_tensor.shape[-2],
        in_periods=args.time_len,
        out_periods=args.pre_time_len,
        batch_size=args.batch_size
    )
    trainer = Trainer(args, model, (train_loader, val_loader), edge_index, loss_fn=torch.nn.MSELoss(), DEVICE=DEVICE)
    trainer.train()
    print("training finished!")

    # Testing ----------------------------------------------------------------------------------------------------------
    model_test = TemporalGNN(
        node_features=test_x_tensor.shape[-2],
        in_periods=args.time_len,
        out_periods=args.pre_time_len,
        batch_size=args.test_batch_size
    )
    param_path = glob(os.path.join(args.log_dir, "*.pth"))[0]
    model_test.load_state_dict(torch.load(param_path))
    print(f"model loaded at {param_path}")
    tester = Tester(args, model_test, test_loader, edge_index, torch.nn.MSELoss(), DEVICE)
    tester.test()
    print("testing finished!")


if __name__ == '__main__':
    input_args = parse()
    main(input_args)
