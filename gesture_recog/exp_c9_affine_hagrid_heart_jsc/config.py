from easydict import EasyDict as edict

opt = edict()
############ postprocessing ###########
opt.postprocessing = edict()
opt.postprocessing.type = "identity"  # add identity scale

opt.postprocessing.add = edict()
opt.postprocessing.add.m = 0.5

opt.postprocessing.identity = edict()

opt.postprocessing.scale = edict()
opt.postprocessing.scale.m = -1  # m < n
opt.postprocessing.scale.n = 1

############ loss setting ############
opt.loss = edict()
opt.loss.alpha = 1
############ fea loss setting ############
opt.fea_loss = edict()
opt.fea_loss.fea_loss_type = "ce"  # mse_loss or ce or arcface or arcface_ce

# mse_loss
opt.fea_loss.mse_loss = edict()
opt.fea_loss.mse_loss.alpha = 64
# wing_loss
opt.fea_loss.wing_loss = edict()
opt.fea_loss.wing_loss.width = 6
opt.fea_loss.wing_loss.curvature = 0.5
# arcface
opt.fea_loss.arcface = edict()
opt.fea_loss.arcface.emb_size = 128
opt.fea_loss.arcface.num_class = 14
opt.fea_loss.arcface.scale = 30
opt.fea_loss.arcface.margin = 0.5
############ base setting ############
opt.base = edict()
opt.base.seed = 1234
opt.base.checkpoint = "/zhoudu/checkpoints/gesture_recog"
opt.base.checkpoint_freq_in_iter = None
opt.base.checkpoint_freq_in_epoch = 10
opt.base.display_freq = 20
opt.base.test_freq_in_epoch = 5
opt.base.test_freq_in_iter = None
############ optimizer setting ############
opt.optim = edict()
opt.optim.max_epoch = 20
opt.optim.optim_type = "adamw"

opt.optim.adam = edict()
opt.optim.adam.lr = 1e-2
opt.optim.adam.lr_betas = (0.5, 0.999)
opt.optim.adam.lr_gamma = 0.1
opt.optim.adam.lr_steps = [150, 180]
opt.optim.adam.weight_decay = 1e-5

opt.optim.adamw = edict()
opt.optim.adamw.lr = 1e-3
opt.optim.adamw.lr_betas = (0.5, 0.999)
opt.optim.adamw.lr_gamma = 0.1
opt.optim.adamw.lr_steps = [60, 80]
opt.optim.adamw.weight_decay = 1e-2

opt.optim.sgd = edict()
opt.optim.sgd.lr = 0.08
opt.optim.sgd.lr_gamma = 0.1
opt.optim.sgd.lr_steps = [15, 25, 35]
opt.optim.sgd.momentum = 0.9
opt.optim.sgd.weight_decay = 1e-3

opt.optim.warmup = edict()
opt.optim.warmup.enable = False
opt.optim.warmup.num_epoch = 0.1  # epoch num for warmup stage, decimal is supported.
opt.optim.warmup.strategy = "cos"  # cos | linear | fix
opt.optim.warmup.init_lr = 1e-5

opt.optim.amp_level = "O2"  # "O0": pure FP32, "O1": mix, "O2": almost fp16 mix, "O3": pure fp16

############# data setting ################
opt.data = edict()
opt.data.num_workers = 4  # num_workers for DataLoader
opt.data.input_shape = (3, 224, 224)
opt.data.class_num = 9
opt.data.garbage_class_num = 0
opt.data.use_real_img = False
opt.data.is_self_supervised = False
opt.data.bias = 127.5
opt.data.scale = 0.0078125
opt.data.need_image = False
opt.data.trainset = ["common1", "common2", "common3", "indoor_multi_scenario", "indoor_office", "hagrid", "jsc220713"
                     ]
opt.data.train_bs = [256]
# opt.data.trainset = [ "common",
#                      ]
# opt.data.train_bs = [128]

opt.data.testset = "common_test"
opt.data.test_bs = 100

############# type ################
opt.data.kpt_type = "regression"  # regression or heatmap
opt.data.debug = False
opt.data.heatmap = edict()
opt.data.heatmap.normalize_kpt = True
opt.data.heatmap.heatmap_size = (64, 64)
opt.data.heatmap.sigma = 5
opt.data.heatmap.radius = 3
opt.data.heatmap.use_expand_bbox = True
opt.data.heatmap.expand_ratio = (2, 2)

############# data trans ################
opt.data.trans = edict()
# expand_bbox
opt.data.trans.expand_bbox = edict()
opt.data.trans.expand_bbox.inacc_box = 0.0
opt.data.trans.expand_bbox.ratio = None
opt.data.trans.expand_bbox.ratio_min = 0.9
opt.data.trans.expand_bbox.ratio_max = 1.1
opt.data.trans.expand_bbox.seed = 1234
# crop_padding
opt.data.trans.crop_padding = edict()
opt.data.trans.crop_padding.output_shape = opt.data.input_shape
opt.data.trans.crop_padding.scale_box = 1.0
opt.data.trans.crop_padding.add_padding = False
opt.data.trans.crop_padding.keep_ratio = False
opt.data.trans.crop_padding.seed = 1234
# rotate
opt.data.trans.rotate = edict()
opt.data.trans.rotate.p = 0.3
opt.data.trans.rotate.min_angle = -30
opt.data.trans.rotate.max_angle = 30
opt.data.trans.rotate.class_weight = [0.0, 0.0,
                                      0.0, 0.0,
                                      1.0, 0.0,
                                      0.0, 0.0]
opt.data.trans.rotate.seed = 1234
# trunc
opt.data.trans.trunc = edict()
opt.data.trans.trunc.p = 0.3
opt.data.trans.trunc.start_ratio = 0.0
opt.data.trans.trunc.end_ratio = 0.4
opt.data.trans.trunc.num = 2
opt.data.trans.trunc.p2 = 0.2
opt.data.trans.trunc.must_cut_p = 0.5
opt.data.trans.trunc.seed = 1234
# rotate_or_trunc
opt.data.trans.rotate_or_trunc = edict()
opt.data.trans.rotate_or_trunc.p = 0.3
opt.data.trans.rotate_or_trunc.min_angle = -30
opt.data.trans.rotate_or_trunc.max_angle = 30
opt.data.trans.rotate_or_trunc.class_weight = [0.1, 0.1,
                                               0.05, 0.05,
                                               0.05, 0.05,
                                               0.3, 0.3]
opt.data.trans.rotate_or_trunc.start_ratio = 0.0
opt.data.trans.rotate_or_trunc.end_ratio = 0.4
opt.data.trans.rotate_or_trunc.num = 2
opt.data.trans.rotate_or_trunc.p2 = 0.2
opt.data.trans.rotate_or_trunc.must_cut_p = 0.5
opt.data.trans.rotate_or_trunc.seed = 1234
# truncv2
opt.data.trans.truncv2 = edict()
opt.data.trans.truncv2.p = 0.3
opt.data.trans.truncv2.class_weight = [
    0.2, 0.2, 0.2, 0.2,
    0.01, 0.01, 0.01, 0.01,
    0.03, 0.03, 0.03, 0.03,
    0.01, 0.01, 0.01, 0.01,
]
opt.data.trans.truncv2.seed = 1234
# color
opt.data.trans.color = edict()
opt.data.trans.color.p = 0.5
opt.data.trans.color.scope = 0.125
opt.data.trans.color.seed = 1234
opt.data.trans.color.mode = 2
# gray
opt.data.trans.gray = edict()
opt.data.trans.gray.p = 0.2
opt.data.trans.gray.scope = 0.125
opt.data.trans.gray.seed = 1234
opt.data.trans.gray.mode = 2
# color_jitter
opt.data.trans.color_jitter = edict()
opt.data.trans.color_jitter.strength = 1
opt.data.trans.color_jitter.p = 0.5
opt.data.trans.color_jitter.seed = 1
# random_gray
opt.data.trans.random_gray = edict()
opt.data.trans.random_gray.p = 0.1
opt.data.trans.random_gray.seed = 1234
# gaussian_blur
opt.data.trans.gaussian_blur = edict()
opt.data.trans.gaussian_blur.kernel_size = 3
opt.data.trans.gaussian_blur.p = 0.1
opt.data.trans.gaussian_blur.seed = 1234
# affine
opt.data.trans.affine = edict()
opt.data.trans.affine.degrees = 90
opt.data.trans.affine.translate = None
opt.data.trans.affine.scale = None
opt.data.trans.affine.shear = None
opt.data.trans.affine.p = 0.2
opt.data.trans.affine.seed = 1234
############# reader setting ################
opt.reader = edict()
opt.reader.starbox = edict()
opt.reader.starbox.host = "10.128.34.10"
opt.reader.starbox.port = 80
opt.reader.starbox.access_id = "UyticBh3"
opt.reader.starbox.access_key = "NZmwD7nomiRZ"

############# dataset setting ################
opt.dataset = edict()

############# model setting ################
opt.model = edict()
opt.model.mixup_alpha = 0.0


def add_dataset(name, source, bs, sampler="list", num_faceid_field=2, has_id=True, method=None, reader="disk",
                type="face", augments_type=[], is_contain_imgwh=False):
    opt.dataset[name] = edict()
    opt.dataset[name].source = source
    opt.dataset[name].num_faceid_field = num_faceid_field
    opt.dataset[name].has_id = has_id
    opt.dataset[name].method = method
    opt.dataset[name].batch_size = bs
    opt.dataset[name].reader = reader
    opt.dataset[name].type = type
    opt.dataset[name].sampler = sampler
    opt.dataset[name].sampler = sampler
    opt.dataset[name].augments_type = augments_type
    opt.dataset[name].is_contain_imgwh = is_contain_imgwh


########################### Train Dataset ##############################
cls_name = "c9"
add_dataset("common1",
            "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_cloudwalk.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter", ],
            is_contain_imgwh=True)  # num=6707
add_dataset("common2",
            "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1015_1017.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter"],
            is_contain_imgwh=True)  # num=6119
add_dataset("common3",
            "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1018_1019.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter"],
            is_contain_imgwh=True)  # num=6364
add_dataset("indoor_multi_scenario",
            "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/train.indoor_multi_scenario.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter"],
            is_contain_imgwh=True)  # num=6776
add_dataset("indoor_office",
            "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/" + cls_name + "/train.indoor_office.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter"],
            is_contain_imgwh=True)  # num=1074
add_dataset("jsc220713",
            "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/train.jsc220713.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter"],
            is_contain_imgwh=True)  # num=1074
add_dataset("hagrid",
            "/dataset/dataset/ssd/gesture/hagrid/" + cls_name + "/train.hagrid.heart.map.box.match.txt",
            2, has_id=False, reader="disk",
            augments_type=["rotate", "crop_padding", "affine", "gaussian_blur", "color_jitter"],
            is_contain_imgwh=True)  # num=1074
add_dataset("common_neg",
            "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.train.ges30_cloudwalk.ges30_imgs_1015_1017.map.box.txt",
            2, has_id=False, reader="disk", augments_type=["expand_bbox", "rotate", "crop_padding", "gaussian_blur"],
            is_contain_imgwh=True)  # num=20173

########################### Val Dataset ##############################
add_dataset("common_test",
            "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/test.ges30_imgs_1018_1019.map.box.match.txt",
            64, has_id=False, reader="disk", augments_type=[""], is_contain_imgwh=True)  # num=480
add_dataset("t", "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/tiny.txt", 64,
            has_id=False, reader="disk", augments_type=[""], is_contain_imgwh=True)  # num=480
