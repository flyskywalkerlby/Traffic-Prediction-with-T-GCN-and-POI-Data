import os
import sys
import time

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import json

DIR_Qtr = '../data/Q-Traffic Dataset'
DIR_INFO = '../data/Q-Traffic Dataset/data_info'
F_RII = DIR_INFO + '/rii.csv'
F_ROADS_INFO = DIR_INFO + '/roads_info.csv'
F_SPD_NET = DIR_INFO + '/spd_net.csv'
F_R2VV = DIR_INFO + '/r2vv.json'
F_V2GPS = DIR_INFO + '/v2gps.json'
F_POI = DIR_INFO + '/poi.json'
F_POI_POINTS_ALL = DIR_INFO + '/poi_points.csv'
F_PRE = DIR_INFO + '/predictions_mse=0.1266_.npy'

D_FIG = '../pictures'

label_roads_info = ['road', 'length', 'width', 'lon', 'lat', 'snodeid', 'enodeid']

SCALE = 10

# MAX_T_ALL = 5856
T_PRE0 = 4685
MAX_T_PRE = 1171
MAX_T = 5
NumRoad = 12702
NumCross = 11807
GPS_RANGE = (116.3389120052928, 116.44846000529279, 39.89189319509181, 39.969887195091815)

ColorBar = ['brown', 'red', 'orange', 'green']
NumInterval = 3
Spd2ColorStep = 10
Spd0 = 10

COLOR_POI = ['red', 'blue', 'green', 'black', 'white', 'cyan', 'magenta', 'yellow']
NUM_POI = 3
F_POI_POINTS = DIR_INFO + '/poi_points_{}.csv'.format(NUM_POI)


def compress_map_from_gps(scale=SCALE):
    start_t = time.time()

    # load
    print('Load gps and compress...')
    f_gps = DIR_Qtr + '/road_network_sub-dataset/link_gps.v2'
    gps = pd.read_table(f_gps, header=None)
    gps.columns = ['road', 'lon', 'lat']
    num_road = gps.shape[0]

    gps_lon = gps['lon']
    gps_lat = gps['lat']
    # print(np.min(gps_lon), np.max(gps_lon), np.mean(gps_lon))
    # print(np.min(gps_lat), np.max(gps_lat), np.mean(gps_lat))
    '''
        116.100694 116.648434 116.39368600529279
        39.748302 40.138272 39.93089019509181
    '''

    # compress gps range
    rg_ori = (np.min(gps_lon), np.max(gps_lon), np.min(gps_lat), np.max(gps_lat))
    print(rg_ori)

    mean_lon = np.mean(gps_lon)
    rg_lon = np.max(gps_lon) - np.min(gps_lon)
    min_lon = mean_lon - rg_lon / scale
    max_lon = mean_lon + rg_lon / scale
    mean_lat = np.mean(gps_lat)
    rg_lat = np.max(gps_lat) - np.min(gps_lat)
    min_lat = mean_lat - rg_lat / scale
    max_lat = mean_lat + rg_lat / scale

    rg_scale = (min_lon, max_lon, min_lat, max_lat)
    print('GPS_RANGE =', rg_scale)

    # get roads in range
    rii = []
    roads_idx = []
    for i in range(num_road):
        if rg_scale[0] < gps_lon[i] < rg_scale[1] and rg_scale[2] < gps_lat[i] < rg_scale[3]:
            roads_idx.append(i)
            rii.append((gps.at[i, 'road'], len(roads_idx), i))

    # save roads_idx
    print('Save roads_idx...')
    print('NumRoad =', len(roads_idx))
    cs = ['road', 'i', 'i_ori']
    df_rii = pd.DataFrame(rii, columns=cs)
    df_rii.to_csv(F_RII, columns=cs, index=False)

    # save road-gps
    print('Save road2gps...')
    compress_road2gps = pd.DataFrame(gps, index=roads_idx)

    # compress roads_info file
    print('Compress roads_info file...')
    f_roads_info = DIR_Qtr + '/road_network_sub-dataset/road_network_sub-dataset.v2'
    roads_info = pd.read_table(f_roads_info, header=0)
    cs = ['link_id', 'width', 'snodeid', 'enodeid', 'length']
    compress_roads_info = pd.DataFrame(roads_info, columns=cs, index=roads_idx)
    compress_roads_info.columns = ['road', 'width', 'snodeid', 'enodeid', 'length']

    # Merge gps to roads_info
    print('Merge gps to roads_info...')
    merge_roads_info = pd.merge(compress_road2gps, compress_roads_info, sort=False)
    merge_roads_info = merge_roads_info[label_roads_info]
    merge_roads_info.to_csv(F_ROADS_INFO, columns=label_roads_info, index=False)

    # Compress t-spd_pre file
    print('Compress t-spd_pre file...')
    f_spd_net = DIR_Qtr + '/traffic_speed_sub-dataset/traffic_speed_net.csv'
    t = time.time()
    spd_net = pd.read_csv(f_spd_net, header=0)
    print('Spend T =', time.time() - t)
    cs = spd_net.columns.values
    cs = cs[-MAX_T_PRE:-MAX_T_PRE+MAX_T]
    compress_spd_net = pd.DataFrame(spd_net, columns=cs, index=roads_idx)
    cs = compress_spd_net.columns = list(range(MAX_T))
    compress_spd_net.to_csv(F_SPD_NET, columns=cs, index=False)

    print('Done!')
    print('Spend T =', time.time() - start_t)


def generate_vertex2gps():
    roads_info = pd.read_csv(F_ROADS_INFO, header=0)

    # print(roads_info.shape)
    # print(roads_info.columns.values)
    # print(roads_info.loc[0])

    es_id = list(roads_info['road'])
    vvs = list(zip(roads_info['snodeid'], roads_info['enodeid']))
    gps_l = list(zip(roads_info['lon'], roads_info['lat']))

    v2ls = {}
    for i in range(NumRoad):
        gps = gps_l[i]
        for v in vvs[i]:
            try:
                v2ls[v].append(gps)
            except KeyError:
                v2ls[v] = [gps]
    vs_id = list(v2ls.keys())

    nv = len(vs_id)
    print('NumV =', nv)
    vs = list(range(nv))
    es = [(vs_id.index(v1_id), vs_id.index(v2_id)) for v1_id, v2_id in vvs]

    print('vs =', len(vs))
    print('es =', len(es))

    r2vv = dict(zip(es_id, es))
    with open(F_R2VV, 'w') as f:
        json.dump(r2vv, f, indent=4)

    v2gps = {vs_id.index(k): tuple(np.mean(v, axis=0)) for k, v in v2ls.items()}
    with open(F_V2GPS, 'w') as f:
        json.dump(v2gps, f, indent=4)


def compress_poi_file():
    f_poi = DIR_Qtr + '/new/raw_poi_data.json'
    with open(f_poi, 'r', encoding='utf-8') as f:
        poi_raw = json.load(f)

    road2poi = {}
    for road in poi_raw:
        road_id = road['road_id']
        road2poi[road_id] = []
        for poi in road['pois']:
            road2poi[road_id].append((poi['typecode'][:2], poi['location']))

    with open(F_POI, 'w') as f:
        json.dump(road2poi, f, indent=4)


def generate_poi_points():
    with open(F_POI, 'r') as f:
        road2poi = json.load(f)

    label2type = {'05': 0, '06': 1, '08': 2, '09': 3, '10': 4, '14': 5, '15': 6, '17': 7}

    gps_type = set()
    for road, poi_s in road2poi.items():
        c = 0
        for poi in poi_s:
            try:
                poi_type = label2type[poi[0]]
            except KeyError:
                continue
            ll_str = poi[1].split(',')
            lon = float(ll_str[0])
            lat = float(ll_str[1])
            gps_type.add((lon, lat, poi_type, c))
            c += 1
            if c > NUM_POI:
                break
    poi_points = list(gps_type)
    print('NumPOI =', len(poi_points))

    cs = ['lon', 'lat', 'type', 'ord']
    df_poi_points = pd.DataFrame(poi_points, columns=cs)
    df_poi_points.to_csv(F_POI_POINTS, columns=cs, index=False)


def spd_to_color(spd_net, t=0):
    try:
        spd_t = spd_net[str(t)]
    except IndexError:
        spd_t = spd_net

    color_t = []
    for spd in spd_t:
        for i in range(NumInterval):
            if spd < Spd0 + i * Spd2ColorStep:
                color_t.append(ColorBar[i])
        else:
            color_t.append(ColorBar[NumInterval])

    return color_t


def transform_ll(ll):
    h = int(ll)
    ll = (ll - h) * 60
    m = int(ll)
    ll = (ll - m) * 60
    s = round(ll, 1)
    return h, m, s


def my_save_fig(file_path):
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)


def draw(t=0):
    spd_net = pd.read_csv(F_SPD_NET, header=0)
    colors_e = spd_to_color(spd_net, t)

    roads_info = pd.read_csv(F_ROADS_INFO, header=0)

    with open(F_V2GPS, 'r') as f:
        v2gps = json.load(f)
    v2gps = {int(k): v for k, v in v2gps.items()}

    with open(F_R2VV, 'r') as f:
        r2vv = json.load(f)
    es = [tuple(v) for v in r2vv.values()]

    ws = list(roads_info['width'])
    std_w = np.std(ws)
    ws = [w/std_w for w in ws]

    net = nx.Graph()
    # nx.draw_networkx_nodes(net, v2gps, node_size=10)
    nx.draw_networkx_edges(net, v2gps, es, width=ws, edge_color=colors_e)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Traffic Speed GroundTruth')
    plt.xlim(GPS_RANGE[0], GPS_RANGE[1])
    plt.ylim(GPS_RANGE[2], GPS_RANGE[3])

    # my_save_fig(D_FIG + '/gt_{}.png'.format(t+T_PRE0))
    my_save_fig(D_FIG + '/gt.png')

    # plt.show()


def draw_pre(t=0):
    spd_net = np.load(F_PRE)
    spd_net = spd_net[:, t]
    colors_e = spd_to_color(spd_net, t)

    roads_info = pd.read_csv(F_ROADS_INFO, header=0)

    with open(F_V2GPS, 'r') as f:
        v2gps = json.load(f)
    v2gps = {int(k): v for k, v in v2gps.items()}

    with open(F_R2VV, 'r') as f:
        r2vv = json.load(f)
    es = [tuple(v) for v in r2vv.values()]

    ws = list(roads_info['width'])
    std_w = np.std(ws)
    ws = [w / std_w for w in ws]

    net = nx.Graph()
    # nx.draw_networkx_nodes(net, v2gps, node_size=10)
    nx.draw_networkx_edges(net, v2gps, edgelist=es, width=ws, edge_color=colors_e)

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Traffic Speed Prediction')
    plt.xlim(GPS_RANGE[0], GPS_RANGE[1])
    plt.ylim(GPS_RANGE[2], GPS_RANGE[3])

    # my_save_fig(D_FIG + '/pre_{}.png'.format(t+T_PRE0))
    my_save_fig(D_FIG + '/pre.png')


def draw_add_poi():
    spd_net = pd.read_csv(F_SPD_NET, header=0)
    colors_e = spd_to_color(spd_net)

    roads_info = pd.read_csv(F_ROADS_INFO, header=0)

    with open(F_V2GPS, 'r') as f:
        v2gps = json.load(f)
    v2gps = {int(k): v for k, v in v2gps.items()}

    colors_v = []
    size_v = []
    gps_type = pd.read_csv(F_POI_POINTS, header=0)
    for i, tp in enumerate(gps_type['type']):
        v2gps[NumCross+i] = (gps_type.at[i, 'lon'], (gps_type.at[i, 'lat']))
        colors_v.append(COLOR_POI[tp])
        size_v.append(2 * (NUM_POI - gps_type.at[i, 'ord']))
    ns = list(range(len(colors_v)))

    with open(F_R2VV, 'r') as f:
        r2vv = json.load(f)
    es = [tuple(v) for v in r2vv.values()]

    ws = list(roads_info['width'])
    std_w = np.std(ws)
    ws = [w/std_w for w in ws]

    net = nx.Graph()
    # net.add_nodes_from(ns)

    # print(net.nodes)
    # nx.draw_networkx_nodes(net, v2gps, nodelist=ns, node_size=size_v, node_color='cornflowerblue', alpha=0.5)
    nx.draw_networkx_nodes(net, v2gps, nodelist=ns, node_size=size_v, node_color=colors_v, alpha=0.5)
    nx.draw_networkx_edges(net, v2gps, edgelist=es, width=ws, edge_color='black')

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Beijing Roads Network and POI')
    plt.xlim(GPS_RANGE[0], GPS_RANGE[1])
    plt.ylim(GPS_RANGE[2], GPS_RANGE[3])

    my_save_fig(D_FIG + '/road_network_poi.png')

    # plt.show()


if __name__ == '__main__':
    # compress_map_from_gps()
    # generate_vertex2gps()

    # compress_poi_file()
    # generate_poi_points()

    try:
        tt = int(sys.argv[1])
    except (IndexError, ValueError):
        tt = 0

    draw(tt)
    draw_pre(tt)
    draw_add_poi()




