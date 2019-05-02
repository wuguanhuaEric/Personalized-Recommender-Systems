#-*-coding:utf8-*-
"""
date:2018****
get graph from user data
"""
import os
def get_graph_from_data(input_file):
    """
    :param input_file:item rating file
    :return: a dict:{User A:{itemb:1, itemc:1}, itemb:{UserA:1}}
    """
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    graph = {}
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(",")
        if len(item) < 3:
            continue
        userid, itemid, rating = item[0], "item_" + item[1], item[2]
        if float(rating) <score_thr:
            continue
        if userid not in graph:
            graph[userid] = {}
        graph[userid][itemid] = 1
        if itemid not in graph:
            graph[itemid] = {}
        graph[itemid][userid] = 1
    fp.close()
    return graph

def get_item_info(input_file):
    """
    get item info:[title,genre]
    :param input_file:item info file
    :return: a dict key itemid, value:[title, genre]
    """
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    linenum = 0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:
            continue
        elif len(item) == 3:
            itemid, title, genre = item[0], item[1], item[2]
        elif len(item) > 3:
            itemid = item[0]
            genre = item[-1]
            title = ",".join(item[1:-1])
        item_info[itemid] = [title, genre]
    fp.close()
    return item_info

if __name__ == "__main__":
    print(get_graph_from_data("../data/log.txt"))