import argparse
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
import os
from data.dataset_multiview import Gaze_Dataset_Multiview_CrossView
from utils.file_utils import load_pretrained_weights
from pytorch_lightning.loggers import WandbLogger
from model.model_transformer import Transformer_fov_cat_CrossView

def train(args):
    
    print(args)
    base_dir = args.base_dir
    img_out_size = args.image_size                                                          
    head_out_shape = (224, 224)
    hm_size = (64, 64)
    train_dataset = Gaze_Dataset_Multiview_CrossView(base_dir, img_out_size, head_out_shape, hm_size, eval_scene=args.test_scene, test=False)
    val_dataset = Gaze_Dataset_Multiview_CrossView(base_dir, img_out_size, head_out_shape, hm_size, eval_scene=args.test_scene, test=True)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")     
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8)

    setting_name = f'model_{args.model}_test{args.test_scene}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_sample{args.epipolar_sample}_alpha{args.alpha}_beta{args.beta}_dir{args.dir_weight}_fov{args.fov_thres}_var{args.use_var}'
    os.makedirs(os.path.join('./logs', args.project_name), exist_ok=True)
    log_path = os.path.join('./logs', args.project_name, setting_name+'.log')
    
    
    model = Transformer_fov_cat_CrossView(args.lr, image_size=img_out_size, alpha=args.alpha, beta=args.beta, dir_weight=args.dir_weight,
                                fov_thres=args.fov_thres, use_var=args.use_var, hm_size=hm_size, sample_num=args.epipolar_sample,
                                use_epi_attn=args.use_epi_attn, sim_type=args.simtype)
    # load the weights from multi-view setting
    load_pretrained_weights(model, weight_path = args.init_weights)
    
    if len(args.remark)>0:
        setting_name = args.remark + '_' + setting_name
    ckpt_folder = os.path.join(args.ckpt_dir, args.project_name, setting_name)
    checkpoint_callback = ModelCheckpoint(ckpt_folder, filename='{epoch}', save_top_k=-1, every_n_epochs=args.save_every)
    
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)    
    
    wandb_logger = WandbLogger(name=setting_name, project=args.project_name) 
    # initialize from pretrained weights?
    save_ckpt = False if args.debug else True
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        benchmark=True,
        min_epochs=5,
        max_epochs=opt.epochs,
        devices=args.device,
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=True,
        enable_checkpointing=save_ckpt,
        num_sanity_val_steps=1
    )
    
    
    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=20, help='in the unit of dual views, so the real batch size is 40')
    parser.add_argument("--image_size", type=int, nargs='+', required=True, help='input size of image to the model')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--remark", default='')
    parser.add_argument("--model", default='ours')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for AdamW')
    parser.add_argument('--not_use_var', dest='use_var', action='store_false')  
    parser.add_argument("--test_scene", type=str, default='', help='scene for validation')
    #parser.add_argument("--gaze_estimator_weights", default='/data/add_disk0/qiaomu/ckpts/gaze/multiview_gaze/gaze_estimator/subject_level_lr0.0001_coef2.0_bs32/epoch=53.ckpt')
    parser.add_argument("--gaze_estimator_weights", default='/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/gaze360/gaze360_res18_backbone.pt')
    parser.add_argument("--init_weights", default='/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/videoatttarget_init/initial_weights_for_temporal_training.pt')
    parser.add_argument("--project_name", default='Multiview_CrossScene_pretrained')
    parser.add_argument('--fov_thres', type=float, default=0.9, help='threshold for fov heatmap in ChildPlay')
    parser.add_argument('--alpha', type=float, default=10.0, help='weight for heatmap loss')
    parser.add_argument('--beta', type=float, default=0.05, help='weight for visibility loss')
    parser.add_argument('--dir_weight', type=float, default=0.1, help='weight for gaze direction loss')
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--no_epi_attn", action='store_false', dest='use_epi_attn')
    parser.add_argument("--simtype", type=str, default='softmax')
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument('--base_dir', default='/data/add_disk0/qiaomu/datasets/gaze/Multiview_Gaze')
    parser.add_argument("--ckpt_dir", type=str,
                        default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze")
    parser.add_argument('--epipolar_sample', type=int, default=48)
    parser.add_argument("--device", nargs='*', type=int, default=1)
    opt = parser.parse_args()
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    train(opt)