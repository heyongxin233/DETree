import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_embedding import TextEmbeddingModel

class Tree():
    def __init__(self,path):
        self.name = {}
        self.childs = {}
        self.father = {}
        self.dep = {}
        self.root = None
        self.max_dep = 0
        self.subtree = {}
        self.grad_fa = {} # the node closest to the root for each leaf
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                assert len(parts) == 3, "Each line must have exactly three parts"
                
                now,fa,name = parts
                now,fa = int(now),int(fa)
                if name != 'none':
                    self.name[now] = name.split(',')
                if fa != -1:
                    self.childs[fa] = self.childs.get(fa, []) + [now]
                    self.father[now] = fa
                else:
                    self.root = now
        self.fa_pos = torch.zeros((len(self.father),len(self.father)),dtype=torch.bool)

        self.dfs(self.root)
        #max_dep,N,N+K 0/1
        self.pos_down2up = torch.zeros((self.max_dep,len(self.name),len(self.father)),dtype=torch.bool)
        self.neg_down2up = torch.zeros((self.max_dep,len(self.name),len(self.father)),dtype=torch.bool)

        self.pos_up2down = torch.zeros((self.max_dep,len(self.name),len(self.father)),dtype=torch.bool)
        self.neg_up2down = torch.zeros((self.max_dep,len(self.name),len(self.father)),dtype=torch.bool)

        self.pos_center = torch.zeros((self.max_dep,len(self.name)),dtype=torch.long)
        self.mask_center = torch.zeros((self.max_dep,len(self.name),len(self.father)),dtype=torch.bool)

        #max_dep,N 0/1
        self.mask = torch.zeros((self.max_dep,len(self.name)),dtype=torch.bool)
        self.depth = torch.zeros(len(self.name))
        self.labels = torch.zeros(len(self.name),dtype=torch.long)
        self.vis_leaf()
        label_value = list(set(self.grad_fa.values()))
        for key, value in self.grad_fa.items():
            self.labels[key] = label_value.index(value)

    def dfs(self, node, depth=0,grfa=-1):
        self.dep[node] = depth
        self.max_dep = max(self.max_dep, depth)
        if node!=self.root:
            self.subtree[node] = torch.zeros(len(self.father),dtype=torch.bool)
            self.subtree[node][node] = 1

        # if self.fa_pos.get(node) is None:
            if self.father[node] != self.root:
                self.fa_pos[node] = self.fa_pos[self.father[node]].clone()
            self.fa_pos[node][node] = 1
            if grfa == -1:
                grfa = node
        if self.childs.get(node) is None:
            self.grad_fa[node] = grfa
        for child in self.childs.get(node, []):
            self.dfs(child, depth + 1,grfa)
            if node!=self.root:
                self.subtree[node] = torch.logical_or(self.subtree[node], self.subtree[child])

    def gen_leaf_item(self,node):
        last_node = -1
        leaf_id = node
        self.depth[node] = self.dep[node]
        while node != self.root:
            now_dep=self.dep[node]-1
            self.mask[now_dep,leaf_id] = 1
            self.pos_center[now_dep,leaf_id] = node
            self.mask_center[now_dep,leaf_id] = torch.logical_not(torch.logical_or(self.fa_pos[node],self.subtree[node]))
            self.mask_center[now_dep,leaf_id,node] = 1
            if last_node == -1:
                self.pos_down2up[now_dep,leaf_id] = self.subtree[node]
            else:
                self.pos_down2up[now_dep,leaf_id]=torch.logical_xor(self.subtree[node],self.subtree[last_node])
            self.neg_down2up[now_dep,leaf_id]=torch.logical_not(self.subtree[node])

            if self.father[node] == self.root:
                self.neg_up2down[now_dep,leaf_id] = torch.logical_not(self.subtree[node])
            else:
                self.neg_up2down[now_dep,leaf_id] = torch.logical_xor(self.subtree[node],self.subtree[self.father[node]])
            self.pos_up2down[now_dep,leaf_id] = self.subtree[node]

            last_node = node
            node = self.father[node]

    def vis_leaf(self):
        for node, name in self.name.items():
            self.gen_leaf_item(node)


    def display(self):
        for node, name in self.name.items():
            depth = self.dep[node]
            print(f"{depth}- {name} {self.father[node]}")
                
class SimCLR_Tree(nn.Module):
    def __init__(self, opt, fabric):
        super(SimCLR_Tree, self).__init__()

        self.temperature = opt.temperature
        self.opt = opt
        self.fabric = fabric

        adapter_path = getattr(opt, "adapter_path", None)
        self.model = TextEmbeddingModel(
            opt.model_name,
            lora=opt.lora,
            use_pooling=opt.pooling,
            lora_r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout,
            adapter_path=adapter_path,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tree = Tree(opt.tree_txt)

        self.pos_down2up = self.tree.pos_down2up.to(self.device)
        self.neg_down2up = self.tree.neg_down2up.to(self.device)
        self.pos_up2down = self.tree.pos_up2down.to(self.device)
        self.neg_up2down = self.tree.neg_up2down.to(self.device)
        self.pos_center = self.tree.pos_center.to(self.device)
        self.mask_center = self.tree.mask_center.to(self.device)

        self.K = self.pos_down2up.shape[0]

        self.mask = self.tree.mask.to(self.device)
        self.depth = self.tree.depth.to(self.device)
        self.root_labels = self.tree.labels.to(self.device)
        self.esp = torch.tensor(1e-6, device=self.device)
        self.max_dep = self.tree.max_dep
        self.leaf_cnt = len(self.tree.name)

        self.names2id = {}
        for key, value in self.tree.name.items():
            for item in value:
                self.names2id[item] = key

        self.vitual_center = nn.Parameter(
            torch.randn((len(self.tree.father), opt.projection_size), device=self.device),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.vitual_center)
        self.center_labels = torch.arange(len(self.tree.father), dtype=torch.long, device=self.device)
        if adapter_path is not None:
            self.load_tree_state(adapter_path)


    def get_encoder(self):
        return self.model

    def save_pretrained(self, save_directory: str, save_tokenizer: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory, save_tokenizer=save_tokenizer)
        torch.save(
            {"vitual_center": self.vitual_center.detach().cpu()},
            os.path.join(save_directory, "tree_state.pt"),
        )

    def load_tree_state(self, directory: str):
        state_path = os.path.join(directory, "tree_state.pt")
        if not os.path.exists(state_path):
            return
        state = torch.load(state_path, map_location=self.vitual_center.device)
        self.vitual_center.data.copy_(state["vitual_center"].to(self.vitual_center.device))

    def load_from_directory(self, directory: str, is_trainable: bool = True):
        if getattr(self.opt, "lora", False):
            self.model.load_adapter(directory, is_trainable=is_trainable)
        else:
            self.model = TextEmbeddingModel(
                directory,
                lora=False,
                use_pooling=self.opt.pooling,
                output_hidden_states=False,
            )
        self.load_tree_state(directory)

    def _compute_logits(self, q,q_labels,k,k_labels,pos_mask,neg_mask):
        def cosine_similarity_matrix(q, k):
            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        
        def gen_label_mask(relation_matrix,q_labels, k_labels):

            N1 = q_labels.shape[0]
            N2 = k_labels.shape[0] 

            q_labels_expanded = q_labels.unsqueeze(1).expand(-1, N2)  # N1 x N2
            k_labels_expanded = k_labels.unsqueeze(0).expand(N1, -1)  # N1 x N2

            result_matrix = relation_matrix[:,q_labels_expanded, k_labels_expanded]

            return result_matrix

        logits=cosine_similarity_matrix(q,k)
        logits=logits/self.temperature
        logits = logits.unsqueeze(0).expand(self.K,-1,-1) #K,N1,N2

        pos_mask = gen_label_mask(pos_mask,q_labels, k_labels)
        neg_mask = gen_label_mask(neg_mask,q_labels, k_labels) #K,N1,N2

        pos_logits = torch.sum(logits*pos_mask,dim=-1)/torch.max(torch.sum(pos_mask,dim=-1),self.esp)#K,N1
        pos_logits = pos_logits.unsqueeze(-1)#K,N1,1
        neg_logits = logits*neg_mask#K,N1,N2

        logits = torch.cat((pos_logits, neg_logits), dim=-1)#K,N1,N2+1

        #model:model set
        # pos_logits_model = torch.sum(logits*same_model,dim=1)/torch.max(torch.sum(same_model,dim=1),self.esp)# N
        # neg_logits_model=logits*torch.logical_not(same_model)# N,N+K
        # logits_model=torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1)

        return logits

    def forward(self, encoded_batch, labels):
        q = self.model(encoded_batch)
        N1 = q.shape[0]
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_labels = self.fabric.all_gather(labels).view(-1)
        
        now_depth = self.depth[labels].unsqueeze(0).expand(self.K,-1)
        now_mask = self.mask[:,labels]
        # leaf_labels = self.root_labels[labels]

        k = torch.concat((k,self.vitual_center),dim=0)
        k_labels = torch.concat((k_labels,self.center_labels),dim=0)

        logits_sample = self._compute_logits(q,labels,k,k_labels,self.pos_down2up,self.neg_down2up)#K,N1,N2+1
        gt_sample = torch.zeros(logits_sample.shape[:-1], dtype=torch.long,device=logits_sample.device)
        logits_sample = logits_sample.permute(0,2,1)
        loss_smaple1 = F.cross_entropy(logits_sample, gt_sample, reduction='none') #K,N1
        loss_smaple1 = torch.sum((loss_smaple1/now_depth)*now_mask)/N1*self.max_dep

        # out = self.root_classfier(q)
        # loss_classfiy = F.cross_entropy(out, leaf_labels)

        loss = loss_smaple1

        return loss,loss_smaple1

    # def forward(self, encoded_batch, labels):
    #     q = self.model(encoded_batch)
    #     # N1 = q.shape[0]
    #     # k = q.clone().detach()
    #     # k = self.fabric.all_gather(k).view(-1, k.size(1))
    #     # k_labels = self.fabric.all_gather(labels).view(-1)
        
    #     # now_depth = self.depth[labels].unsqueeze(0).expand(self.K,-1)
    #     # now_mask = self.mask[:,labels]
    #     leaf_labels = self.root_labels[labels]

    #     # k = torch.concat((k,self.vitual_center),dim=0)
    #     # k_labels = torch.concat((k_labels,self.center_labels),dim=0)

    #     # logits_sample = self._compute_logits(q,labels,k,k_labels,self.pos_down2up,self.neg_down2up)#K,N1,N2+1
    #     # gt_sample = torch.zeros(logits_sample.shape[:-1], dtype=torch.long,device=logits_sample.device)
    #     # logits_sample = logits_sample.permute(0,2,1)
    #     # loss_smaple1 = F.cross_entropy(logits_sample, gt_sample, reduction='none') #K,N1
    #     # loss_smaple1 = torch.sum((loss_smaple1/now_depth)*now_mask)/N1*self.max_dep

    #     out = self.root_classfier(q)
    #     loss_classfiy = F.cross_entropy(out, leaf_labels)

    #     loss = loss_classfiy

    #     return loss,loss_classfiy

