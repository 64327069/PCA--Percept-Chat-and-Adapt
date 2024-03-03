def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'TinyVirat':
        cfg.data_folder = '/data0/TinyVIRAT_V2/ori_videos'
        cfg.train_annotations = '/data0/TinyVIRAT_V2/tiny_train_528.json'
        cfg.val_annotations = '/data0/TinyVIRAT_V2/tiny_val_v2.json'
        cfg.class_map = '/data0/TinyVIRAT_V2/class_map.json'
        cfg.num_classes = 26
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg

