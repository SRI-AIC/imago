from imago_prev.models.sequential.dsv.dis_seq_vae import DisSeqVAE, elbo_loss

import torch
import os
from absl import app, flags, logging
from tqdm import tqdm
from imago_prev.models.model_util import save_checkpoint, load_checkpoint, checkpoint_exists
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS

flags.DEFINE_string("data_root", "data/sequential", "Path to the train/test directories")
flags.DEFINE_float('learning_rate', 2e-4, 'Optimizer learning rate')
#flags.DEFINE_float('learning_rate', 1e-3, 'Optimizer learning rate')
flags.DEFINE_integer('batch_size', 64, 'Data batch size')
flags.DEFINE_integer('num_epochs', 10000, 'Number of training epochs')
flags.DEFINE_bool('shuffle', True, 'Shuffle data')
flags.DEFINE_integer('num_workers', 4, 'Number of workers for DataLoaders')
flags.DEFINE_string('device', 'cuda:0', "Device to load to")
flags.DEFINE_integer('save_epochs', 10, 'Save interval between epochs')
flags.DEFINE_string('checkpoint_dir', 'output/checkpoints', 'Directory to save checkpoints')
flags.DEFINE_string('tb_logdir', 'output/logs', 'Directory to save Tensorboard logs')
flags.DEFINE_string("model_name", "model", "Model name for this run")


class SpriteDataset(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size;

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fpath = os.path.join(self.path, '%d.sprite' % (idx + 1))
        if not (os.path.isfile(fpath)):
            fpath = os.path.join(self.path, '%d.sprite ' % (idx + 1))
        return torch.load(fpath)


class DSVTrainer:
    def __init__(self, data_root,
                 device="cpu", start_epoch=0, epochs=100, learning_rate=1e-3):
        self.device = device
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch + epochs
        self.learning_rate=learning_rate
        self.model = DisSeqVAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.train_dataset = SpriteDataset(os.path.join(data_root, "train"), 272)
        self.test_dataset = SpriteDataset(os.path.join(data_root, "test"), 5)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=FLAGS.batch_size,
                                                  shuffle=FLAGS.shuffle, num_workers=FLAGS.num_workers)
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=FLAGS.batch_size,
                                                  shuffle=FLAGS.shuffle, num_workers=FLAGS.num_workers)
        self.tb_logdir = FLAGS.tb_logdir
        self.checkpoint_dir = FLAGS.checkpoint_dir
        self.model_name = FLAGS.model_name
        # TODO ensure model is initialized with good settings

    def train(self, resume=True):
        steps = 0
        if checkpoint_exists(self.model_name, self.checkpoint_dir) and resume:
            logging.info("Resuming from checkpoint = {}".format(self.checkpoint_dir))
            steps = load_checkpoint(os.path.join(self.checkpoint_dir, self.model_name), self.model, self.optimizer,
                            device=self.device)
            logging.info("Resuming from step = {}".format(steps))
        self.tb_sw = SummaryWriter(self.tb_logdir)
        for epoch in range(self.start_epoch, self.end_epoch):
            logging.info("\nEpoch: {}".format(epoch))
            tqdm_iter = tqdm(self.trainloader, total=len(self.trainloader))
            loss_total = 0.
            batches_seen = 0
            for data in tqdm_iter:
                data = data.to(self.device)
                steps += data.shape[0]
                self.optimizer.zero_grad()
                res = self.model(data)
                loss, mse, kld = elbo_loss(data, res)
                loss_val, mse_val, kld_val = loss.item(), mse.item(), kld.item()
                self.tb_sw.add_scalar("Loss", loss_val / data.shape[0], steps)
                self.tb_sw.add_scalar("MSE", mse_val / data.shape[0], steps)
                self.tb_sw.add_scalar("KLD", kld_val / data.shape[0], steps)
                loss_total += loss_val
                batches_seen += 1
                tqdm_iter.set_description("Loss={:.5f}, mean={:.5f}, Steps={}".format(loss_val,
                                                                                      loss_total / batches_seen, steps))
                loss.backward()
                self.optimizer.step()
            val_tqdm_iter = tqdm(self.testloader, total=len(self.testloader))
            val_loss_total = 0.
            val_batches_seen = 0

            for val_data in val_tqdm_iter:
                val_data = val_data.to(self.device)
                val_res = self.model(val_data)
                val_loss, val_mse, val_kld = elbo_loss(val_data, val_res)
                val_loss_val, val_mse_val, val_kld_val = val_loss.item(), val_mse.item(), val_kld.item()
                self.tb_sw.add_scalar("Val_Loss", val_loss_val / val_data.shape[0], steps)
                self.tb_sw.add_scalar("Val_MSE", val_mse_val / val_data.shape[0], steps)
                self.tb_sw.add_scalar("Val_KLD", val_kld_val / val_data.shape[0], steps)
                val_loss_total += val_loss
                val_batches_seen += 1
                val_tqdm_iter.set_description("Val Loss={:.5f}, mean={:.5f}".format(val_loss, 
                                                                                    val_loss_total / val_batches_seen))
            # Take image
            combined_img = torch.cat((val_data[0], val_res.Xhat[0]), dim=0)
            combined_img = combined_img.view(16,3,64,64)
            self.tb_sw.add_images("Recon", combined_img, steps)
                
            if epoch % FLAGS.save_epochs == 0:
                logging.info("Checkpointing to {}".format(self.checkpoint_dir))
                save_checkpoint(self.model_name, self.model, self.optimizer, self.checkpoint_dir, step=steps)
                
def main(argv):
    trainer = DSVTrainer(FLAGS.data_root,
                         device=FLAGS.device, epochs=FLAGS.num_epochs, learning_rate=FLAGS.learning_rate)
    trainer.train()

if __name__ == "__main__":
    app.run(main)