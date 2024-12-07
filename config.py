name = "baseline"
path = 'lip_imgs_112' # video path
test = True
batch_size = 32
gpus = 4
lr = 3e-4
num_workers = 2
max_epoch = 100
pre_train_path="pretrain/checkpoints-epoch=22-train_loss=0.57-train_wer=0.07-val_loss=5.08-val_wer=0.73.ckpt"
version = 0