python eval_multiview_condition.py --device 1 --test_scene Kitchen --model transformer --project_name Eval_Multiview --batch_size 40 --image_size 512 384 --init_weights /nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze/multiview_ckpts/Kitchen.ckpt
python eval_multiview_condition.py --device 1 --test_scene Commons --model transformer --project_name Eval_Multiview --batch_size 40 --image_size 512 384 --init_weights /nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze/multiview_ckpts/Commons.ckpt
python eval_multiview_condition.py --device 1 --test_scene Lab --model transformer --project_name Eval_Multiview --batch_size 40 --image_size 512 384 --init_weights /nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze/multiview_ckpts/Lab.ckpt
python eval_multiview_condition.py --device 1 --test_scene Shop --model transformer --project_name Eval_Multiview --batch_size 40 --image_size 512 384 --init_weights /nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze/multiview_ckpts/Shop.ckpt

