import os, sys

from absl import logging
from sklearn import model_selection
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import collections

from imago_prev.models.behav.reaver_behav import REAVER_ACT_SPECS, ReaverBehavModel, scan
from imago_prev.models.behav.losses import vae_loss, behavior_losses, JSD
from imago_prev.data.datasets import IDPDataset
from imago_prev.models.semframe.frames import make_category_lookup, CategoricalFrame, NumericFrame
from sc2recorder.utils import ensure_numpy, ensure_torch, ensure_torch_long, set_random_seed

import pprint

def evaluate(rb_model, test_dataloader,  dist_loss_type=JSD, device="cpu", beta=1):
    """
    Evaluates the given model against the dataset, using the given logit loss type.
    Returns the significant loss (stuff we care about), a dict of named mean losses,
    and then the loss object itself for backwards()
    """
    rb_model.behav_model.eval()
    rb_model.obs_model.eval()
    pp = pprint.PrettyPrinter(indent=4)
    named_losses = collections.OrderedDict()
    correct_by_act = collections.OrderedDict()
    for datum in tqdm(test_dataloader):
        datum = ensure_torch_rec(device, datum)
        Os, Vs = datum[0], datum[1]
        Bs = datum[2:]

        loss, _ = vae_loss(rb_model.obs_model, Os, named_losses=named_losses, beta=beta)
        behav_loss, _, _ = behavior_losses(rb_model.behav_model, Os, Bs, Vs, dist_loss_type, 
                                     named_losses=named_losses, correct_by_act=correct_by_act, device=device)
        loss += behav_loss

    sig_loss = 0.
    for _, loss in named_losses.items():
        sig_loss += loss
        
    for act_spec, num_correct in correct_by_act.items():
        micro_acc = num_correct / (test_dataloader.batch_size * len(test_dataloader))
        named_losses["microAcc_{}".format(act_spec)] = micro_acc
        sig_loss += 1.0 - micro_acc

    rb_model.behav_model.train()
    rb_model.obs_model.train()
    pp.pprint(named_losses)
    return sig_loss, named_losses, loss


def _record(tb_sw, named_losses, prefix, step):
    for name, losses in named_losses.items():
        mean_loss = np.mean(losses)
        tb_sw.add_scalar("{}_{}".format(prefix, name), mean_loss, step)


def train(output_dir, train_dataloader, test_dataloader, 
          reaver_act_spec,
          dist_loss_type=JSD,
          n_hidden=128, device="cpu", epochs=100, beta=1):
    components = assemble_default_obs_components()
    rb_model = ReaverBehavModel(output_dir, components, reaver_act_spec=reaver_act_spec,
                                device=device, n_hidden=n_hidden).to(device)
    print(rb_model)
    print(rb_model.behav_model)
    optimizer = torch.optim.Adam(rb_model.parameters())

    if rb_model.model_save_exists():
        step = rb_model.load_model()
    else:
        step = 0
    tb_logdir = os.path.join(output_dir, "tb_logs")
    tb_sw = SummaryWriter(tb_logdir)
    logging.info("Saving TB logs to {}".format(tb_sw))

    def _validate():
        combined_train_images = rb_model.sample_images(train_dataloader)
        combined_val_images = rb_model.sample_images(test_dataloader)
        tb_sw.add_image("Train Images Sample", combined_train_images, step)
        tb_sw.add_image("Val Images Sample", combined_val_images, step)
        sig_loss, eval_named_losses, _ = evaluate(rb_model, test_dataloader, device=device, beta=beta)
        _record(tb_sw, eval_named_losses, "Val", step)
        return sig_loss
    
    rb_model.train()
    _validate()
    for epoch in range(epochs):
        tqdm_iter = tqdm(train_dataloader)
        for datum in tqdm_iter:
            datum = ensure_torch_rec(device, datum)
            Os, Vs = datum[0], datum[1]
            Bs = datum[2:]
            batch_size = len(Os)
            step += batch_size
            optimizer.zero_grad()
            named_losses = collections.OrderedDict()
            correct_by_act = collections.OrderedDict()
            loss, _ = vae_loss(rb_model.obs_model, Os, named_losses=named_losses, beta=beta)
            recon_loss = loss.item()
            if False:
                behav_loss = 0.
            else:
                behav_loss, _, _ = behavior_losses(rb_model.behav_model, Os, Bs, Vs, dist_loss_type, 
                                             named_losses=named_losses, correct_by_act=correct_by_act,
                                             device=device)
                loss += behav_loss
                behav_loss = behav_loss.item()
            tqdm_iter.set_description("Recon_loss={:.5f}, behav_loss={:.5f}".format(recon_loss, behav_loss))
            for act_spec, num_correct in correct_by_act.items():
                micro_acc = num_correct / (train_dataloader.batch_size )
                named_losses["microAcc_{}".format(act_spec)] = micro_acc
            _record(tb_sw, named_losses, "Train", step)
            loss.backward()
            optimizer.step()
        print(named_losses)
        epoch_sig_loss = _validate()
        rb_model.save_model(step, epoch_sig_loss)
        _validate()
    return rb_model

