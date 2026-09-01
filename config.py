from torch_geometric.nn import SAGEConv,GCNConv


default_config={
    "n_eighborss" : 15,
    "n_top_genes" : 2000,
    "hidden_dim" : 64,
    "latent_dim" : 64,
    "conv_layer" :SAGEConv,
    "epochs_cluster" : 300,
    "pre_epoch":200,
    "lr_cluster" : 0.001}


configs={
"Adam":{
    "n_eighborss" : 15,
    "n_top_genes" : 2000,
    "hidden_dim" : 64,
    "latent_dim" : 64,
    "conv_layer" :SAGEConv,
    "epochs_cluster" : 300,
    "pre_epoch":200,
    "lr_cluster" : 0.00},
    "Muraro":{
        "n_eighborss" : 15,
        "n_top_genes" : 2000,
        "hidden_dim" : 64,
        "latent_dim" : 64,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 500,
        "pre_epoch":200,
        "lr_cluster" : 0.001},
    "Bach":{
        "n_eighborss" : 15,
        "n_top_genes" : 2000,
        "hidden_dim" : 64,
        "latent_dim" : 64,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 300,
        "pre_epoch":200,
        "lr_cluster" : 0.001},
    "Campbell":default_config,
    "Cao_2020_Spleen":default_config,
    "Quake_10x_Bladder":default_config,
    "Quake_10x_Limb_Muscle":default_config,
    "Quake_10x_Spleen":default_config,
    "Quake_Smart-seq2_Diaphragm":default_config,
    "Quake_Smart-seq2_Limb_Muscle":default_config,
    "Quake_Smart-seq2_Lung":default_config,
    "Quake_Smart-seq2_Trachea":default_config,
    "Romanov":default_config,
    "Shekhar":default_config,
    "Tosches_turtle":default_config,
    "Wang_Large_Intestine":default_config,
    "Young":default_config
    }