from .combined_model import *

import torchvision
from torchvision import transforms
import pdb

"""
TODO: Standardize the values based on train dataloader
"""

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


def compute_beta(step):
    if step <= 100000:
        return 0
    elif step <= 200000:
        return 1e-8
    elif step <= 300000:
        return 1e-7
    elif step <= 400000:
        return 1e-6
    elif step <= 500000:
        return 1e-5
    elif step <= 600000:
        return 1e-4
#    elif step <= 700000:
#        return 1e-3
#    elif step <= 800000:
#        return 1e-2
    return 1e-3  # Cap it at this point

def train(output_dir, train_dataloader, test_dataloader, 
          reaver_act_spec,
          dist_loss_type=JSD,
          components=COMPONENTS,
          device="cpu", epochs=100):
    rb_model = ReaverBehavModel2(output_dir, components, reaver_act_spec=reaver_act_spec,
                                device=device).to(device)
    optimizer = torch.optim.Adam(rb_model.parameters())

    # Compute the value params from the train dataset, then apply to both
    # the train and test data
    v_mean, v_std = train_dataloader.dataset.compute_V_params()
    logging.info("Computed Value function mean={:.5f}, std={:.5f}".format(v_mean, v_std))
    train_dataloader.dataset.set_V_params(v_mean, v_std)
    test_dataloader.dataset.set_V_params(v_mean, v_std)    
    
    if rb_model.model_save_exists():
        step = rb_model.load_model()
    else:
        step = 0
    print(rb_model)
    tb_logdir = os.path.join(output_dir, "tb_logs")
    tb_sw = SummaryWriter(tb_logdir)
    logging.info("Saving TB logs to {}".format(tb_logdir))

    def _validate(beta):
        xform = transforms.ToTensor()
        combined_train_images = rb_model.sample_images(train_dataloader, to_tensor=False)
        combined_val_images = rb_model.sample_images(test_dataloader, to_tensor=False)
        combined_train_images.save(os.path.join(output_dir, "current_train.jpg"))
        combined_val_images.save(os.path.join(output_dir, "current_val.jpg"))
        tb_sw.add_image("Train Images Sample", xform(combined_train_images), step)
        tb_sw.add_image("Val Images Sample", xform(combined_val_images), step)
        sig_loss, eval_named_losses, _ = evaluate(rb_model, test_dataloader, device=device, beta=beta)
        _record(tb_sw, eval_named_losses, "Val", step)
        return sig_loss
    
    #beta = 1e-5  # Starting beta value
    #beta = 0 # Had to set beta to 0 to get red/blue in allegiance.  Tried experiment setting beta to 1 after burning in with beta=0.
    rb_model.train()
    _validate(compute_beta(step))
    for epoch in range(epochs):
        tqdm_iter = tqdm(train_dataloader)
        for datum in tqdm_iter:
            # tune beta based on current step
            beta = compute_beta(step)
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
            if not(LEARN_BEHAVIOR):
                behav_loss = 0.
            else:
                behav_loss, _, _ = behavior_losses(rb_model.behav_model, Os, Bs, Vs, dist_loss_type, 
                                             named_losses=named_losses, correct_by_act=correct_by_act,
                                             device=device)
                loss += behav_loss
                behav_loss = behav_loss.item()
            tqdm_iter.set_description("Recon_loss={:.5f}, behav_loss={:.5f}, beta={:10f}".format(recon_loss, behav_loss, beta))
            for act_spec, num_correct in correct_by_act.items():
                micro_acc = num_correct / (train_dataloader.batch_size )
                named_losses["microAcc_{}".format(act_spec)] = micro_acc
            _record(tb_sw, named_losses, "Train", step)
            loss.backward()
            optimizer.step()
        print("Current named losses in play:")
        print(named_losses)
        epoch_sig_loss = _validate(beta)
        print("Saving model")
        rb_model.save_model(step, epoch_sig_loss)

    return rb_model

