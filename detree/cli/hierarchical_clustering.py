import argparse
import random
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import euclidean, squareform
from sklearn.metrics import silhouette_score

def read_similarity_matrix(file_path: Path):
    with file_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    names = lines[0].strip().split()
    matrix = []

    for line in lines[1:]:
        row = line.strip().split()[1:]
        matrix.append([float(x) for x in row])

    similarity_matrix = np.array(matrix)
    return names, similarity_matrix

class TreeNode:
    def __init__(self, name=None):

        self.name = name
        self.children = []
        self.value = 0
        self.split = True

    def add_child(self, child):
        self.children.append(child)
        
def build_tree(Z, names):
    nodes = [TreeNode(name) for name in names]
    for i, link in enumerate(Z):
        node = TreeNode()
        node.value = link[2]
        node.add_child(int(link[0]))
        node.add_child(int(link[1]))
        nodes.append(node)
    return nodes

def find_best_thold(node_idx,nodes, distance_matrix,min_socre=0,max_socre=1):
    node = nodes[node_idx]
    threshold_range = np.linspace(min_socre * node.value, max_socre * node.value, 50)
    silhouette_scores = []
    all_n_clusters = []

    for threshold in threshold_range:
        labels,_ = gen_label_from_node(node_idx,nodes,threshold)
        labels = sorted(labels,key=lambda x:x[1])
        labels = [x[0] for x in labels]
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and n_clusters < len(distance_matrix):
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:
            score = -1
        silhouette_scores.append(score)
        all_n_clusters.append(n_clusters)
    best_threshold_idx = np.argmax(silhouette_scores)
    best_threshold = threshold_range[best_threshold_idx]
    best_score = silhouette_scores[best_threshold_idx]
    return best_threshold, best_score

def gen_label_from_node(node_idx,nodes,thd,now_label=0):
    node = nodes[node_idx]
    if len(node.children)==0:
        return [(now_label,node_idx)],now_label
    else:
        if node.value>thd:
            label_list = []
            for child in node.children:
                now_label_list,now_label = gen_label_from_node(child,nodes,thd,now_label)
                now_label+=1
                label_list+=now_label_list
            return label_list,now_label
        else:
            label_list = []
            for child in node.children:
                now_label_list,now_label = gen_label_from_node(child,nodes,thd,now_label)
                label_list+=now_label_list
            return label_list,now_label

def find_new_root(node_idx,nodes,thd):
    node = nodes[node_idx]
    if node.value<=thd:
        return [node_idx]

    new_root = []
    for child in node.children:
        new_root+=find_new_root(child,nodes,thd)
    return new_root

def get_leaf(node_idx,nodes):
    node = nodes[node_idx]
    if len(node.children)==0:
        return [node_idx]

    leaf_list = []
    for child in node.children:
        leaf_list+=get_leaf(child,nodes)
    return leaf_list

def merge_tree(node_idx,nodes,distance_matrix,deep=0,end_thd=0.25):
    if len(nodes[node_idx].children)==0:
        return
    print(f"Node {node_idx}: Value: {nodes[node_idx].value}, Depth: {deep}")
    if nodes[node_idx].value<=end_thd or deep>=5:
        nodes[node_idx].children = get_leaf(node_idx,nodes)
        nodes[node_idx].split = False
        return
    leaf_list = np.array(sorted(get_leaf(node_idx,nodes)))
    new_distance_matrix = distance_matrix[leaf_list][:,leaf_list]
    best_threshold, best_score = find_best_thold(node_idx, nodes, new_distance_matrix,min_socre=0)
    if best_score==-1:
        nodes[node_idx].children = get_leaf(node_idx,nodes)
        return
    new_root = find_new_root(node_idx,nodes,best_threshold)
    nodes[node_idx].children = new_root

    for child in new_root:
        merge_tree(child,nodes,distance_matrix,deep=deep+1,end_thd=end_thd)

def merge_dict(a,b):
    for key in b.keys():
        if key in a.keys():
            a[key]+=b[key]
        else:
            a[key] = b[key]
    return a

def update_tree(node_idx, nodes, edge_list, fa=-1, deep=0):
    node = nodes[node_idx]

    if len(node.children)==0:
        edge_list.append((fa,node_idx,[nodes[node_idx].name]))
        return {deep:[[node_idx]]}

    if node.split==False:
        leafs = get_leaf(node_idx,nodes)
        edge_list.append((fa,node_idx,[nodes[idx].name for idx in leafs]))
        return {deep:[leafs]}

    edge_list.append((fa,node_idx,[]))
    new_tree = {}
    for child in node.children:
        new_tree = merge_dict(
            new_tree,
            update_tree(child, nodes, edge_list, node_idx, deep=deep+1),
        )
    if deep not in new_tree.keys():
        new_tree[deep] = []
    new_tree[deep].append(get_leaf(node_idx,nodes))

    return new_tree

def color_distance(c1, c2):
    return euclidean(c1[:3], c2[:3])  # only consider the RGB components

def ensure_color_diversity(colors, min_distance=0.2):
    random.shuffle(colors)
    for i in range(1, len(colors)):
        if color_distance(colors[i], colors[i-1]) < min_distance:
            for j in range(i + 1, len(colors)):
                if color_distance(colors[i], colors[j]) > min_distance:
                    colors[i], colors[j] = colors[j], colors[i]
                    break
    return colors


def draw_table(new_tree, names, max_deep=3, save_path='fig/E/test.pdf'):
    base_list = new_tree[0][0]
    data = [base_list]
    cmap = cm.get_cmap('tab20c', 2048)
    cmap = [cmap(i) for i in range(2048)]
    cmap = ensure_color_diversity(cmap)
    cell_colours = [['#FFDDC1' for _ in base_list]]
    color_start=0

    for i in range(1,max_deep+1):
        if i not in new_tree.keys():
            print(f"Level {i} not in new_tree")
            continue
        data.append([names[base] for base in base_list])
        color_list = []
        for k,base in enumerate(base_list):
            color_id = -1
            for j in range(len(new_tree[i])):
                if base in new_tree[i][j]:
                    color_id = j
                    break
            if color_id==-1:
                color_list.append(cell_colours[-1][k])
            else:
                color_list.append(cmap[color_start+color_id])
        cell_colours.append(color_list)
        color_start+=len(new_tree[i])

    data = list(zip(*data))
    cell_colours = list(zip(*cell_colours))            
    columns = ['Node ID']+['Level {}'.format(i) for i in range(1,max_deep+1)]
    plt.figure(figsize=(30, 40))
    table = plt.table(cellText=data, colLabels=columns, loc='center', cellLoc='center',
                colColours=['#f5f5f5']*len(columns),cellColours=cell_colours)
    table.auto_set_column_width([0, 1]) 
    plt.axis('off')
    plt.savefig(save_path, format='pdf' ,bbox_inches='tight',pad_inches=0.01)

def fix_asymmetry(matrix):
    matrix = (matrix + matrix.T) / 2
    return matrix

def rename(edge):
    cnt=0
    reid={}
    du={}
    edge_dict={}
    queue=[]
    for i in range(len(edge)):
        du[edge[i][0]]=du.get(edge[i][0],0)+1
        edge_dict[edge[i][1]]=edge[i]
        if edge[i][2] != []:
            queue.append(edge[i][1])
    while len(queue)>0:
        now = queue.pop(0)
        if now==-1:
            reid[now]=-1
            continue
        if now not in reid.keys():
            reid[now]=cnt
            cnt+=1
        now_edge = edge_dict[now]
        du[now_edge[0]]-=1
        if du[now_edge[0]]==0:
            queue.append(now_edge[0])
    new_edge = [(reid[x[0]],reid[x[1]],x[2]) for x in edge]
    return new_edge

def save_edge(edge,save_path):
    with open(save_path,'w') as f:
        for e in edge:
            if e[2]:
                name_str = ','.join(e[2])
            else:
                name_str = 'none'
            f.write(f"{e[1]} {e[0]} {name_str}\n")

def filter_class(names, similarity_matrix):
    choose_idx = []
    for i in range(len(names)):
        if 'extend' not in names[i] and 'polish' not in names[i] and\
              'translate' not in names[i] and 'paraphrase' not in names[i]:
            if 'B' in names[i] or 'human' in names[i]:
                choose_idx.append(i)
            else:
                if random.random()<0.3:
                    choose_idx.append(i)
        elif 'human' in names[i]:
            if random.random()<0.3:
                choose_idx.append(i)
        elif random.random()<0.15:
            choose_idx.append(i)
    new_names = [names[i] for i in choose_idx]
    choose_idx  = np.array(choose_idx)
    new_similarity_matrix = similarity_matrix[choose_idx][:,choose_idx]
    return new_names, new_similarity_matrix

def filter(names, similarity_matrix,filter_human=False,filter_llm=False,filter_mix=False):
    choose_idx = []
    for i in range(len(names)):
        if names[i] == 'human' and filter_human:
            continue
        if filter_llm and 'human' not in names[i]:
            continue
        if filter_mix and 'human' in names[i] and names[i]!='human':
            continue
        choose_idx.append(i)
    new_names = [names[i] for i in choose_idx]
    choose_idx  = np.array(choose_idx)
    new_similarity_matrix = similarity_matrix[choose_idx][:,choose_idx]
    return new_names, new_similarity_matrix

def reid_tree_dict(tree_dict, nodes, names):
    name_to_index = {name: idx for idx, name in enumerate(names)}
    for deep,values in tree_dict.items():
        rename_now = []
        # print(values,len(values))
        for list_ in values:
            now_list = []
            for idx in list_:
                name = nodes[idx].name
                if name not in name_to_index:
                    name_to_index[name] = len(names)
                    names.append(name)
                name_idx = name_to_index[name]
                now_list.append(name_idx)
            rename_now.append(now_list)
        tree_dict[deep] = rename_now
    return tree_dict

def gen_tree(similarity_matrix,names,opt):
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance_matrix = squareform(distance_matrix)
    Z = linkage(condensed_distance_matrix, method='weighted')  # alternative methods include 'single', 'complete', or 'ward'
    if opt.save_drg:
        plt.figure(figsize=(30, 47))
        dendrogram(Z, labels=names, orientation='right',leaf_font_size=16)  # rotate the dendrogram so the root is on the right
        plt.savefig(opt.dendrogram_path, format='pdf' ,bbox_inches='tight')
    nodes = build_tree(Z, names)
    merge_tree(len(nodes)-1,nodes,distance_matrix,end_thd=opt.end_score)

    return nodes

def chage_tree_priori1(nodes):
    human_node = TreeNode(name='human')
    root = TreeNode()
    root.add_child(len(nodes))
    root.add_child(len(nodes)-1)
    nodes.append(human_node)
    nodes.append(root)
    return nodes

def chage_tree_priori2(human_nodes,llm_nodes):
    root = TreeNode()
    root.add_child(len(human_nodes)-1)
    root.add_child(len(human_nodes)+len(llm_nodes)-1)
    for i in range(len(llm_nodes)):
        llm_nodes[i].children = [len(human_nodes)+x for x in llm_nodes[i].children]
    nodes = human_nodes+llm_nodes
    nodes.append(root)
    return nodes

def chage_tree_priori3(co_nodes,llm_nodes):
    human_node = TreeNode(name='human')
    root = TreeNode()
    root.add_child(len(co_nodes)+len(llm_nodes))
    root.add_child(len(co_nodes)-1)
    root.add_child(len(co_nodes)+len(llm_nodes)-1)
    for i in range(len(llm_nodes)):
        llm_nodes[i].children = [len(co_nodes)+x for x in llm_nodes[i].children]
    nodes = co_nodes+llm_nodes
    nodes.append(human_node)
    nodes.append(root)
    return nodes

def randmo_filter(names, similarity_matrix):
    choose_idx = []
    for i in range(len(names)):
        if 'human' in names[i]:
                choose_idx.append(i)
        elif 'fair' in names[i] or 'pplm' in names[i] or 'gpt2-pytorch' in names[i] or ' transfo' in names[i]  or 'ctrl' in names[i]:
                continue
        elif 'xlnet' in names[i] or 'grover' in names[i]:
            if random.random()<0.07:
                choose_idx.append(i)
        elif random.random()<0.22:
                choose_idx.append(i)
    new_names = []
    for i in choose_idx:
        if names[i].startswith('7B') or names[i].startswith('13B') or names[i].startswith('30B') or names[i].startswith('65B'):
            new_names.append('LLaMA_'+names[i])
        else:
            new_names.append(names[i])
    choose_idx  = np.array(choose_idx)
    new_similarity_matrix = similarity_matrix[choose_idx][:,choose_idx]
    return new_names, new_similarity_matrix

def ishuman(name):
    return ('human' in name)
def ismachine(name):
    return ('machine' in name or 'rephrase' in name)

def get_llm(x):
    if 'gpt-3.5-turbo' in x:
        return 'gpt-3.5-turbo'
    elif 'gpt-4o' in x:
        return 'gpt-4o'
    elif 'llama-3.3-70b' in x:
        return 'llama-3.3-70b'
    elif 'gemini-1.5-pro' in x:
        return 'gemini-1.5-pro'
    elif 'claude-3-5-sonnet' in x:
        return 'claude-3-5-sonnet'
    elif 'qwen2.5-72b' in x:
        return 'qwen2.5-72b'
    else:
        raise ValueError(f"Invalid class name: {x}")
    
def get_name(name):
    name = name.split('_')
    assert len(name) == 2
    if ishuman(name[0]):
        if name[1]=='humanize:human' or name[1]=='human':
            return 'human'
        elif name[1]=='humanize:tool':
            return 'human_humanize_tool'
        else:
            llm_name = get_llm(name[1])
            return f'human_rephrase_{llm_name}'
    elif ismachine(name[0]):
        llm_name = get_llm(name[0])
        if name[1]=='humanize:human' or name[1]=='human':
            return f'{llm_name}_humanize_human'
        elif name[1]=='humanize:tool':
            return f'{llm_name}_humanize_tool'
        elif 'humanize:' in name[1]:
            llm_name2 = get_llm(name[1])
            return f'{llm_name}_humanize_{llm_name2}'
        else:
            return llm_name

def clear_names(names):
    new_names = []
    for name in names:
        new_names.append(get_name(name))
    return new_names

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct the HAT tree from a similarity matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--file-path', type=Path, required=True, help='Input similarity matrix text file.')
    parser.add_argument('--priori',type=int,default=1,choices=[0,1,2,3])
    parser.add_argument('--save-txt-path', type=Path, required=True, help='Destination path for the tree definition.')
    parser.add_argument('--save-table-path', type=Path, required=True, help='Destination path for the visualised table.')
    parser.add_argument('--dendrogram-path', type=Path, default=None, help='Optional path for the dendrogram PDF when saved.')
    parser.add_argument('--save-drg', action='store_true', help='Persist the dendrogram PDF alongside the tree.')
    parser.add_argument('--no-save-drg', dest='save_drg', action='store_false')
    parser.set_defaults(save_drg=True)
    parser.add_argument('--save-max-dep', type=int, default=5)
    parser.add_argument('--end-score', type=float, default=0.1)
    parser.add_argument('--randmo-filter', action='store_true', help='Randomly subsample similarity entries.')
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    opt = parser.parse_args(argv)

    names, similarity_matrix = read_similarity_matrix(opt.file_path)
    if opt.save_drg:
        if opt.dendrogram_path is None:
            opt.dendrogram_path = opt.save_table_path.with_name(
                f"{opt.save_table_path.stem}_dendrogram.pdf"
            )
        opt.dendrogram_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        opt.dendrogram_path = None
    similarity_matrix = fix_asymmetry(similarity_matrix)
    if opt.randmo_filter:
        names, similarity_matrix = randmo_filter(names, similarity_matrix)
    # names = clear_names(names)
    if opt.priori==1:
        llm_names, llm_similarity_matrix = filter(names, similarity_matrix,filter_human=True)
        nodes = gen_tree(llm_similarity_matrix,llm_names,opt)
        nodes = chage_tree_priori1(nodes)

    elif opt.priori==2:
        human_names, human_similarity_matrix = filter(names, similarity_matrix,filter_llm=True)
        human_nodes = gen_tree(human_similarity_matrix,human_names,opt)
        llm_names, llm_similarity_matrix = filter(names, similarity_matrix,filter_human=True,filter_mix=True)
        llm_nodes = gen_tree(llm_similarity_matrix,llm_names,opt)
        nodes = chage_tree_priori2(human_nodes,llm_nodes)
    
    elif opt.priori==3:
        co_names, co_similarity_matrix = filter(names, similarity_matrix,filter_llm=True,filter_human=True)
        co_nodes = gen_tree(co_similarity_matrix,co_names,opt)
        llm_names, llm_similarity_matrix = filter(names, similarity_matrix,filter_human=True,filter_mix=True)
        llm_nodes = gen_tree(llm_similarity_matrix,llm_names,opt)
        nodes = chage_tree_priori3(co_nodes,llm_nodes)
    
    elif opt.priori==0:
        nodes = gen_tree(similarity_matrix,names,opt)
    else:
        raise ValueError("Invalid value for --priori. Choose from 0, 1, 2, or 3.")

    edge=[]
    tree_dict = update_tree(len(nodes)-1, nodes, edge)
    edge = rename(edge)
    opt.save_txt_path.parent.mkdir(parents=True, exist_ok=True)
    opt.save_table_path.parent.mkdir(parents=True, exist_ok=True)
    save_edge(edge,opt.save_txt_path)
    tree_dict = reid_tree_dict(tree_dict, nodes, names)
    draw_table(tree_dict, names, max_deep=opt.save_max_dep, save_path=opt.save_table_path)


if __name__ == "__main__":
    main()


__all__ = ["build_argument_parser", "main", "read_similarity_matrix", "gen_tree"]
