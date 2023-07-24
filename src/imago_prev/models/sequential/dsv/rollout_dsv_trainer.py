from imago_prev.models.sequential.dsv.rollout_dsv import RolloutDisSeqVAE, loss_fn

import torch
import torchvision
import os
from absl import app, flags, logging
from tqdm import tqdm
from imago_prev.models.model_util import save_checkpoint, load_checkpoint, checkpoint_exists
from torch.utils.tensorboard import SummaryWriter

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("data_root", "data/combat1", "Path to the train/test directories")
#flags.DEFINE_float('learning_rate', 2e-4, 'Optimizer learning rate')
flags.DEFINE_float('learning_rate', 1e-3, 'Optimizer learning rate')
flags.DEFINE_integer('batch_size', 64, 'Data batch size')
flags.DEFINE_integer('num_epochs', 100000, 'Number of training epochs')
flags.DEFINE_bool('shuffle', True, 'Shuffle data')
flags.DEFINE_integer('num_workers', 4, 'Number of workers for DataLoaders')
flags.DEFINE_string('device', 'cuda:0', "Device to load to")
flags.DEFINE_integer('save_epochs', 100, 'Save interval between epochs')
flags.DEFINE_string("output_dir", "output/rollout_dsv/combat1", "models, outputs, and logs stored here")
flags.DEFINE_string("model_name", "rollout_dsv", "Model name for this run")


class SpriteDataset(torch.utils.data.Dataset):
    def __init__(self, path, size=None):
        self.path = path
        if size is None:
            # Scan directory for *.sprite files
            sprite_files = [x for x in os.listdir(self.path) if x.endswith('.sprite')]
            self.length = len(sprite_files)
        else:
            self.length = size;

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fpath = os.path.join(self.path, '%d.sprite' % (idx + 1))
        if not (os.path.isfile(fpath)):
            fpath = os.path.join(self.path, '%d.sprite ' % (idx + 1))
        return torch.load(fpath)


class RolloutDSVTrainer:
    def __init__(self, data_root,
                 device="cpu", start_epoch=0, epochs=100, learning_rate=1e-3):
        self.device = device
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch + epochs
        self.learning_rate = learning_rate
        self.model = RolloutDisSeqVAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.train_dataset = SpriteDataset(os.path.join(data_root, "train"), 272)
        self.test_dataset = SpriteDataset(os.path.join(data_root, "test"), 5)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=FLAGS.batch_size,
                                                       shuffle=FLAGS.shuffle, num_workers=FLAGS.num_workers)
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=FLAGS.batch_size,
                                                      shuffle=FLAGS.shuffle, num_workers=FLAGS.num_workers)
        self.tb_logdir = os.path.join(FLAGS.output_dir, "logs")
        self.checkpoint_dir = os.path.join(FLAGS.output_dir, "checkpoints")
        self.images_dir = os.path.join(FLAGS.output_dir, "images")
        self.model_name = FLAGS.model_name
        os.makedirs(self.images_dir, exist_ok=True)
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
                loss_res = loss_fn(data, res)
                for k,v in loss_res._asdict().items():
                    mean_val = v.item() / data.shape[0]
                    self.tb_sw.add_scalar(k, mean_val, steps)
                loss_val = loss_res.loss.item()
                loss_total += loss_val
                batches_seen += 1
                tqdm_iter.set_description("Loss={:.5f}, mean={:.5f}, Steps={}".format(loss_val,
                                                                                      loss_total / batches_seen, steps))
                loss_res.loss.backward()
                self.optimizer.step()
            val_tqdm_iter = tqdm(self.testloader, total=len(self.testloader))
            val_loss_total = 0.
            val_batches_seen = 0

            for val_data in val_tqdm_iter:
                with torch.no_grad():
                    val_data = val_data.to(self.device)
                    val_res = self.model(val_data)
                    val_loss_res = loss_fn(val_data, val_res)
                    for k,v in val_loss_res._asdict().items():
                        mean_val = v.item() / val_data.shape[0]
                        self.tb_sw.add_scalar("val_{}".format(k), mean_val, steps)
                    val_loss = val_loss_res.loss.item()
                    val_loss_total += val_loss
                    val_batches_seen += 1
                    val_tqdm_iter.set_description("Val Loss={:.5f}, mean={:.5f}".format(val_loss,
                                                                                        val_loss_total / val_batches_seen))


            if epoch % FLAGS.save_epochs == 0:
                logging.info("Checkpointing to {}".format(self.checkpoint_dir))
                save_checkpoint(self.model_name, self.model, self.optimizer, self.checkpoint_dir, step=steps)
                # Take image
                for val_data in self.testloader:
                    with torch.no_grad():
                        val_data = val_data.to(self.device)
                        val_res = self.model(val_data)
                        combined_val_img = torch.cat((val_data[0], val_res.X_hat[0]), dim=0)
                        imgs_fname = os.path.join(self.images_dir,
                                                  "recon_s{}.png".format(steps))
                        torchvision.utils.save_image(combined_val_img, imgs_fname)
                        self.tb_sw.add_images("Recon", combined_val_img, steps)
                        
                        # Rollouts
                        for n in range(1,5):
                            logging.debug("\n\nRunning rollout N={}\n\n".format(n))
                            rollout_result = self.model.sample_rollout(val_data, n=n)
                            rollout_fname = os.path.join(self.images_dir, "rollout_n{}_s{}.png".format(n, steps))
                            X_next_frames = rollout_result.X_next[0]
                            logging.debug("\n\n Saving rollout, X_next_frames.shape={}".format(
                                rollout_result.X_next.shape
                            ))
                            torchvision.utils.save_image(X_next_frames, rollout_fname)
                            self.tb_sw.add_images("Rollout_{}".format(n), X_next_frames, steps)
                        break


def main(argv):
    trainer = RolloutDSVTrainer(FLAGS.data_root,
                                device=FLAGS.device, epochs=FLAGS.num_epochs, learning_rate=FLAGS.learning_rate)
    trainer.train()


if __name__ == "__main__":
    app.run(main)