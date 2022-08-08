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
opt.fea_loss.fea_loss_type = "ce"  # mse_loss or ce

# mse_loss
opt.fea_loss.mse_loss = edict()
opt.fea_loss.mse_loss.alpha = 64
# wing_loss
opt.fea_loss.wing_loss = edict()
opt.fea_loss.wing_loss.width = 6
opt.fea_loss.wing_loss.curvature = 0.5
############ base setting ############
opt.base = edict()
opt.base.seed = 1234
opt.base.checkpoint = '/zhoudu/checkpoints/gesture_recog'
opt.base.checkpoint_freq_in_iter = None
opt.base.checkpoint_freq_in_epoch = 10
opt.base.display_freq = 20
opt.base.test_freq_in_epoch = 10
opt.base.test_freq_in_iter = None
############ optimizer setting ############
opt.optim = edict()
opt.optim.max_epoch = 200
opt.optim.optim_type = 'adamw'

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
opt.optim.adamw.lr_steps = [150, 180]
opt.optim.adamw.weight_decay = 1e-2

opt.optim.sgd = edict()
opt.optim.sgd.lr = 0.08
opt.optim.sgd.lr_gamma = 0.1
opt.optim.sgd.lr_steps = [15, 25, 35]
opt.optim.sgd.momentum = 0.9
opt.optim.sgd.weight_decay = 1e-3

opt.optim.amp_level = "O2"  # "O0": pure FP32, "O1": mix, "O2": almost fp16 mix, "O3": pure fp16

############# data setting ################
opt.data = edict()
opt.data.num_workers = 4  # num_workers for DataLoader
opt.data.input_shape = (3, 64, 64)
opt.data.class_num = 14
opt.data.bias = 127.5
opt.data.scale = 0.0078125
opt.data.need_image = False
opt.data.trainset = [ "common"
                     ]
opt.data.train_bs = [128]
opt.data.testset = "common_test"
opt.data.test_bs = 100

############# kpt type ################
opt.data.kpt_type = "regression"  # regression or heatmap
opt.data.debug = False
opt.data.heatmap = edict()
opt.data.heatmap.normalize_kpt = True
opt.data.heatmap.heatmap_size = (64, 64)
opt.data.heatmap.sigma = 5
opt.data.heatmap.radius = 3
opt.data.heatmap.use_expand_bbox = True
opt.data.heatmap.expand_ratio = (2, 2)

############# expand ################
opt.data.use_expand_bbox = True
opt.data.expand_bbox = edict()
opt.data.expand_bbox.inacc_box = 0.0
opt.data.expand_bbox.ratio = None
opt.data.expand_bbox.seed = 1234

############# crop and padding ################
opt.data.crop_padding = edict()
opt.data.crop_padding.scale_box = 1.0
opt.data.crop_padding.add_padding = False
opt.data.crop_padding.keep_ratio = True
opt.data.crop_padding.seed = 1234

############# data trans ################
opt.data.trans = edict()
opt.data.trans.type = [""]
# rotate
opt.data.trans.rotate = edict()
opt.data.trans.rotate.p = 0.3
opt.data.trans.rotate.min_angle = -30
opt.data.trans.rotate.max_angle = 30
opt.data.trans.rotate.class_weight = [0.1, 0.1,
                                      0.00, 0.00,
                                      0.00, 0.0,
                                      0.4, 0.4]
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
############# reader setting ################
opt.reader = edict()
opt.reader.starbox = edict()
opt.reader.starbox.host = '10.128.34.10'
opt.reader.starbox.port = 80
opt.reader.starbox.access_id = 'UyticBh3'
opt.reader.starbox.access_key = 'NZmwD7nomiRZ'

############# dataset setting ################
opt.dataset = edict()


def add_dataset(name, source, bs, sampler='list', num_faceid_field=2, has_id=True, method=None, reader='disk',
                type='face', augments_type=None, is_contain_imgwh=False):
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
add_dataset('common',
            '/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/train.ges30_cloudwalk.ges30_imgs_1015_1017.map.box.txt',
            2, has_id=False, reader='disk', augments_type=["rotate"], is_contain_imgwh=True)  # num=20173

########################### Val Dataset ##############################
add_dataset('common_test', '/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/test.ges30_imgs_1018_1019.map.box.txt', 64,
            has_id=False, reader='disk', augments_type=[""], is_contain_imgwh=True)  # num=480
add_dataset('t', '/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/tiny.txt', 64,
            has_id=False, reader='disk', augments_type=[""], is_contain_imgwh=True)  # num=480
