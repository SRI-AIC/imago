"""
Simple VAE implementations used in scene reconstruction of multinomial targets.
"""
from sc2recorder.utils import ensure_numpy, b64encode_img

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from IPython.core.debugger import set_trace
import pdb
from PIL import ImageChops

from .model_util import load_checkpoint, save_checkpoint
from .eval import evaluate_model, compute_divergence, reform_semantic_pixarray

def multinomial_loss_fn(y_hat, y_gold):
    """ Given a hypothesis matrix of the shape (batch, TARGETS, width, height) and
    a target of (batch, width, height) with integer labels for TARGET offset, computes
    the softmax and NLL loss.
    """
    hyp = F.log_softmax(y_hat, dim=1)
    loss = F.nll_loss(hyp, y_gold)
    return loss



def _scatter_sanitycheck():
    probs = torch.randn(5, 21, 512, 512)
    print(probs.shape)
    max_idx = torch.argmax(probs, 1, keepdim=True)
    print(max_idx.shape)
    one_hot = torch.FloatTensor(probs.shape)
    one_hot.zero_()
    X1 = one_hot.scatter_(1, max_idx, 1)

    # Sanity check
    X2 = torch.zeros(probs.shape)
    for bidx in range(5):
        for row in tqdm(range(512)):
            for col in range(512):
                idx = max_idx[bidx, 0, row, col]
                X2[bidx, idx, row, col] = 1
    torch.eq(X1, X2)
    return torch.sum(X1 - X2) # Should return 0


def render_z(unit_renderer, model, Z, C):
    """ Given the model, decode the Z and return the image """
    Y_h = model.decode(Z, C)
    Y_h1 = unit_renderer.metadata.unravel(Y_h)
    imgs = unit_renderer.unit_type_display(unit_renderer.metadata.onehot2unit_type(Y_h1))
    return imgs


def render_results(unit_renderer, model, X, C=None, C1=None,
                   device="cpu", disp_width=3, return_html=True, 
                  deterministic=True):
    """ Convenience routine for rendering VAE based results.  By default returns results in HTML
    format, expecting to be part of a 3 column table body.  If not, returns results as a dictionary.
    If C1 is present, we attempt an interpolated rendering.
    """
    if C is not None and model.is_conditional():
        if C1 is not None:
            res = model.forward_interpolate(X, C, C1, deterministic=deterministic)
            if isinstance(res, dict):
                Y_h = res['x']
            else:
                Y_h, Mu, Logvar, Z = res
        else:
            if deterministic:
                res = model.forward_det(X, C)
                if isinstance(res, dict):
                    Y_h = res['x']
                else:
                    Y_h, Mu, Logvar, Z = res
            else:
                Y_h, Mu, Logvar, Z = model.forward(X, C)
        distances = evaluate_model(model, X, C, device=device)
    else:
        Y_h, Mu, Logvar, Z = model.forward(X)
        distances = evaluate_model(model, X, device=device)
    Y_h1 = unit_renderer.metadata.unravel(Y_h)
    X1 = unit_renderer.metadata.unravel(X)
    guess_imgs = unit_renderer.unit_type_display(unit_renderer.metadata.onehot2unit_type(Y_h1))
    gold_imgs = unit_renderer.unit_type_display(unit_renderer.metadata.onehot2unit_type(X1))
    results = []
    idx = 0
    for guess_img, gold_img, distance in zip(guess_imgs, gold_imgs, distances):
        diff_img = ImageChops.subtract_modulo(gold_img, guess_img)
        if return_html:
            html_str = """<tr><td colspan="3" style="text-align:left;"><hr/></td></tr>"""
            html_str += """<tr><td colspan="3" style="text-align:left;">idx={idx}, divergence={divg:.5f}</td></tr>""".format(idx=idx, divg=distance)
            html_str += """<tr><td><b>Guess</b></td><td><b>Gold</b></td><td><b>Diff</b></td></tr>"""

            html_str += """<tr><td><img src="{guess}" /></td><td><img src="{gold}"/></td><td><img src="{diff}"/></td></tr>
            """.format(
                gold=b64encode_img(gold_img),
                guess=b64encode_img(guess_img),
                diff=b64encode_img(diff_img)
            )
            results.append(html_str)
        else:
            results.append({"guess_img": guess_img,
             "gold_img": gold_img,
             "diff_img": diff_img,
             "distance" : distance })
        idx += 1                                                                                     
    return results


def train_model(model, optimizer, train_dataloader, test_dataloader, metadata,
                epochs=10, device="cpu",
                save_model_dir="output/model",
                save_model_epochs=100,
                kl_penalty_regime="full"):
    """ Convenience routines for training VAEs defined here using unit type datasets.
    """
    print("Training Model, KL Penalty Regime={}".format(kl_penalty_regime))
    model.train()
    N_DISTINCT_UNITS = metadata.N_DISTINCT_UNITS
    OBS_WIDTH = metadata.OBS_WIDTH
    OBS_HEIGHT = metadata.OBS_HEIGHT
    steps = 0
    for epoch in range(epochs):
        tqdm_iter = tqdm(train_dataloader)
        epoch_loss = 0.
        for X, Y_gold, C in tqdm_iter:
            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y_gold = torch.tensor(Y_gold, dtype=torch.long).to(device)
            C = torch.tensor(C, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            if model.is_conditional():
                Y_hat, mu, logvar, z = model.forward(X, C)
            else:
                Y_hat, mu, logvar, z = model.forward(X)
            Y_hat = reform_semantic_pixarray(Y_hat, N_DISTINCT_UNITS, OBS_WIDTH, OBS_HEIGHT)
            loss = multinomial_loss_fn(Y_hat, Y_gold)
            if kl_penalty_regime == "no_kl":
                pass
            else:
                # Default, full KL
                loss += torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))  # KL divergence term, closed form
            # TODO: Add KL Divergence for resampling layer
            train_loss = loss.item()
            epoch_loss += train_loss
            tqdm_iter.set_description("Epoch {}, Loss={:.5f}".format(epoch, epoch_loss))
            loss.backward()
            optimizer.step()
            steps += 1
        if save_model_dir and epoch > 0 and (epoch % save_model_epochs == 0):
            print("Saving checkpoint")
            save_checkpoint("model_{}.pt".format(epoch), model, optimizer, save_model_dir, steps)
        with torch.no_grad():
            model.eval()
            val_loss_total = 0.
            for val_X, val_Y, val_C in test_dataloader:
                val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
                val_Y = torch.tensor(val_Y, dtype=torch.long).to(device)
                val_C = torch.tensor(val_C, dtype=torch.float32).to(device)
                if model.is_conditional():
                    Y_hat, mu, logvar, z = model.forward(val_X, val_C)
                else:
                    Y_hat, mu, logvar, z = model.forward(val_X)
                Y_hat = reform_semantic_pixarray(Y_hat, N_DISTINCT_UNITS, OBS_WIDTH, OBS_HEIGHT)
                val_loss_total += multinomial_loss_fn(Y_hat, val_Y).item()
            print("Epoch {} Mean validation loss={}".format(epoch, val_loss_total/len(train_dataloader)))
            compute_divergence(model, test_dataloader, device=device)
            print("- - - - - - - - - -")
            model.train()
    if save_model_dir:
        save_checkpoint("model_{}.pt".format(epoch), model, optimizer, save_model_dir, steps)
    model.eval()


class VAE(nn.Module):
    def __init__(self, N_INPUT, N_HIDDEN, N_LATENT):
        super(VAE, self).__init__()
        self.N_INPUT, self.N_LATENT, self.N_HIDDEN =  N_INPUT, N_LATENT, N_HIDDEN
        self.fc1  = nn.Linear(N_INPUT, N_HIDDEN)
        self.fc21 = nn.Linear(N_HIDDEN, N_LATENT)
        self.fc22 = nn.Linear(N_HIDDEN, N_LATENT)
        self.fc3  = nn.Linear(N_LATENT, N_HIDDEN)
        self.fc4  = nn.Linear(N_HIDDEN, N_INPUT)

    def is_conditional(self): 
        return False
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps*std + mu
            return z
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.N_INPUT))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z



class CondVAE(nn.Module):
    def __init__(self, N_INPUT, N_COND, N_LATENT, N_HIDDEN):
        super(CondVAE, self).__init__()
        self.N_INPUT, self.N_COND, self.N_LATENT, self.N_HIDDEN =  N_INPUT, N_COND, N_LATENT, N_HIDDEN
        self.fc1  = nn.Linear(N_INPUT + N_COND, N_HIDDEN)
        self.fc21 = nn.Linear(N_HIDDEN, N_LATENT)
        self.fc22 = nn.Linear(N_HIDDEN, N_LATENT)
        self.fc3  = nn.Linear(N_LATENT + N_COND, N_HIDDEN)
        self.fc4  = nn.Linear(N_HIDDEN, N_INPUT)

    def is_conditional(self): 
        return True
        
    def encode(self, x, c):
        x = x.view(-1, self.N_INPUT)
        i = torch.cat((x, c), 1)
        h1 = F.relu(self.fc1(i))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps*std + mu
            return z
        else:
            return mu

    def decode(self, z, c):
        z1 = torch.cat((z, c), 1)
        h3 = F.relu(self.fc3(z1))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar, z

    def forward_det(self, x, c):
        mu, logvar = self.encode(x, c)
        z = mu
        return self.decode(z, c), mu, logvar, z

    def forward_interpolate(self, x, c, c1, deterministic=False):
        """ Encode with X,C but decode with X, C1"""
        mu, logvar = self.encode(x, c)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z, c1), mu, logvar, z

    def make_input(self, x, c):
        x = x.view(-1, self.N_INPUT)
        i = torch.cat((x, c), 1)
        h1 = F.relu(self.fc1(i))
        return h1

    
class Unflatten2D(nn.Module):
    def __init__(self, channel_size, h_dim=1, w_dim=1):
        super(Unflatten2D, self).__init__()
        self.channel_size = channel_size
        self.h_dim, self.w_dim = h_dim, w_dim
        
    def forward(self, x):
        return x.view(-1, self.channel_size, self.h_dim, self.w_dim)


class ConvVAE(nn.Module):
    def __init__(self, N_TARGETS, N_LATENT=128, N_HIDDEN=32):
        super(ConvVAE, self).__init__()
        self.WIDTH = 64
        self.N_TARGETS = N_TARGETS
        self.N_LATENT, self.N_HIDDEN = N_LATENT, N_HIDDEN

        self.encoder = nn.Sequential(
            nn.Conv2d(N_TARGETS, N_HIDDEN, kernel_size=4, stride=2),  # H = 31
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN, N_HIDDEN * 2, kernel_size=4, stride=2),  # H = 14
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 2, N_HIDDEN * 4, kernel_size=4, stride=2),  # H = 6
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 4, N_HIDDEN * 8, kernel_size=4, stride=2),  # H = 2
            nn.ReLU(),
            nn.Flatten()
        )

        h = N_HIDDEN * 8 * 2 ** 2
        self.fc21 = nn.Linear(h, N_LATENT)  # mu
        self.fc22 = nn.Linear(h, N_LATENT)  # logvar

        self.fc3 = nn.Linear(N_LATENT, h)  # Decode first layer
        self.decoder = nn.Sequential(
            Unflatten2D(h, 1, 1),
            nn.ConvTranspose2d(h, N_HIDDEN * 4, kernel_size=5, stride=2),  # 5 x 5
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN * 4, N_HIDDEN * 2, kernel_size=5, stride=2),  # 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN * 2, N_HIDDEN, kernel_size=6, stride=2),  # 30 x 30
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN, N_TARGETS, kernel_size=6, stride=2),  # 64 x 64
        )

    def is_conditional(self):
        return False

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z
        else:
            return mu

    def decode(self, z):
        h3 = self.fc3(z)
        return self.decoder(h3)

    def embed(self, x, reparameterize=False):
        mu, logvar = self.encode(x)
        if reparameterize:
            return self.reparameterize(mu, logvar)
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def forward_det(self, x):
        """ Deterministic, with no resampling (just use means)
        """
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(z), mu, logvar, z


    
class ConvCondVAE(nn.Module):
    def __init__(self, N_COND, N_TARGETS, N_LATENT=128, N_HIDDEN=32):
        super(ConvCondVAE, self).__init__()
        self.WIDTH = 64
        self.N_TARGETS = N_TARGETS
        self.N_COND, self.N_LATENT, self.N_HIDDEN = N_COND, N_LATENT, N_HIDDEN
        
        self.encoder = nn.Sequential(
            nn.Conv2d(N_TARGETS + N_COND, N_HIDDEN, kernel_size=4, stride=2), # H = 31
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN, N_HIDDEN * 2, kernel_size=4, stride=2), # H = 14
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 2, N_HIDDEN * 4, kernel_size=4, stride=2), # H = 6
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 4, N_HIDDEN * 8, kernel_size=4, stride=2), # H = 2
            nn.ReLU(),
            nn.Flatten()
        )
        
        h = N_HIDDEN * 8 * 2**2
        self.fc21 = nn.Linear(h, N_LATENT)  # mu
        self.fc22 = nn.Linear(h, N_LATENT)  # logvar

        self.fc3  = nn.Linear(N_LATENT + N_COND, h) # Decode first layer
        self.decoder = nn.Sequential(
            Unflatten2D(h, 1, 1),
            nn.ConvTranspose2d(h, N_HIDDEN * 4, kernel_size=5, stride=2), # 5 x 5
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN * 4, N_HIDDEN * 2, kernel_size=5, stride=2), # 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN * 2, N_HIDDEN, kernel_size=6, stride=2), # 30 x 30
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN, N_TARGETS, kernel_size=6, stride=2), # 64 x 64
        )

    def is_conditional(self):
        return True
        
    def encode(self, x, C):
        # Take the input C and repeat it along each of the image input channels
        c1 = C.view(-1, C.shape[1], 1, 1).repeat(1, 1, self.WIDTH, self.WIDTH)
        #pdb.set_trace()
        x1 = torch.cat((x, c1), 1)
        h1 = self.encoder(x1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps*std + mu
            return z
        else:
            return mu

    def decode(self, z, c):
        z1 = torch.cat((z, c), 1)
        h3 = self.fc3(z1)
        return self.decoder(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar, z
    
    def embed(self, x, c, reparameterize=False):
        mu, logvar = self.encode(x, c)
        if reparameterize:
            return self.reparameterize(mu, logvar)
        else:
            return mu
    
    def forward_det(self, x, c):
        """ Deterministic, with no resampling (just use means)
        """
        mu, logvar = self.encode(x, c)
        z = mu
        return self.decode(z, c), mu, logvar, z

    def forward_interpolate(self, x, c, c1, deterministic=False):
        """ Encode with X,C but decode with X, C1"""
        mu, logvar = self.encode(x, c)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z, c1), mu, logvar, z
    
    
    
class ConvCondVAEv2(nn.Module):
    """ 2nd variant, where conditioning information is concatenated to result of convolutional encoding
    """
    def __init__(self, N_COND, N_TARGETS, N_LATENT=128, N_HIDDEN=32):
        super(ConvCondVAEv2, self).__init__()
        self.WIDTH = 64
        self.N_TARGETS = N_TARGETS
        self.N_COND, self.N_LATENT, self.N_HIDDEN = N_COND, N_LATENT, N_HIDDEN
        
        self.encoder = nn.Sequential(
            nn.Conv2d(N_TARGETS, N_HIDDEN, kernel_size=4, stride=2), # H = 31
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN, N_HIDDEN * 2, kernel_size=4, stride=2), # H = 14
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 2, N_HIDDEN * 4, kernel_size=4, stride=2), # H = 6
            nn.ReLU(),
            nn.Conv2d(N_HIDDEN * 4, N_HIDDEN * 8, kernel_size=4, stride=2), # H = 2
            nn.ReLU(),
            nn.Flatten()
        )
        
        h = N_HIDDEN * 8 * 2**2
        self.fc21 = nn.Linear(h + N_COND, N_LATENT)  # mu
        self.fc22 = nn.Linear(h + N_COND, N_LATENT)  # logvar

        self.fc3  = nn.Linear(N_LATENT + N_COND, h) # Decode first layer
        self.decoder = nn.Sequential(
            Unflatten2D(h, 1, 1),
            nn.ConvTranspose2d(h, N_HIDDEN * 4, kernel_size=5, stride=2), # 5 x 5
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN * 4, N_HIDDEN * 2, kernel_size=5, stride=2), # 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN * 2, N_HIDDEN, kernel_size=6, stride=2), # 30 x 30
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN, N_TARGETS, kernel_size=6, stride=2), # 64 x 64
        )

    def is_conditional(self):
        return True
        
    def encode(self, x, C):
        # Take the input C and repeat it along each of the image input channels
        #c1 = C.view(-1, C.shape[1], 1, 1).repeat(1, 1, self.WIDTH, self.WIDTH)
        #pdb.set_trace()
        #x1 = torch.cat((x, c1), 1)
        h1 = self.encoder(x)
        h1_c = torch.cat((h1, C), 1)
        return self.fc21(h1_c), self.fc22(h1_c)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps*std + mu
            return z
        else:
            return mu

    #import pdb
    def decode(self, z, c):
        #if z.shape[0] != c.shape[0]:
        # Weird, happens when we filter trajectories
        #pdb.set_trace()
        z1 = torch.cat((z, c), 1)
        h3 = self.fc3(z1)
        return self.decoder(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar, z
    
    def embed(self, x, c, reparameterize=False):
        mu, logvar = self.encode(x, c)
        if reparameterize:
            return self.reparameterize(mu, logvar)
        else:
            return mu
    
    def forward_det(self, x, c):
        """ Deterministic, with no resampling (just use means)
        """
        mu, logvar = self.encode(x, c)
        z = mu
        return self.decode(z, c), mu, logvar, z

    def forward_interpolate(self, x, c, c1, deterministic=False):
        """ Encode with X,C but decode with X, C1"""
        mu, logvar = self.encode(x, c)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z, c1), mu, logvar, z

