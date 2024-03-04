def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'TinyVirat':
        cfg.data_folder = './json/ori_videos'
        cfg.train_annotations = './json/tiny_train_528.json'
        cfg.val_annotations = './json/tiny_val_v2.json'
        cfg.class_map = './json/class_map.json'
        cfg.num_classes = 26
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg

