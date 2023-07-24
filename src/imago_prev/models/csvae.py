import torch
from torch import nn
from .multinomial_vaes import Unflatten2D, reform_semantic_pixarray, multinomial_loss_fn
from .eval import evaluate_model, compute_divergence
from .model_util import save_checkpoint
from tqdm import tqdm
import pdb

from scipy.stats import entropy
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
from IPython.core.debugger import set_trace
import pdb
from PIL import ImageChops
from sc2recorder.utils import ensure_numpy, b64encode_img

def gauss_KL_divg(mu1, logvar1, mu2=None, logvar2=None):
    """
    Computes the KL divergence for two Gaussians, parameterized by their mean and
    logvars.  If mu2 and logvar2 are not defined, then uses a N(0,1) as the target
    :param mu1:
    :param logvar1:
    :param mu2:
    :param logvar2:
    :return:
    """
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.ones_like(logvar1)
    return torch.mean(0.5*(logvar2 - logvar1) +\
                     (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 0.5)



def render_results(unit_renderer, model, X, C, C1=None,
                   device="cpu", disp_width=3, return_html=True, 
                  deterministic=True):
    """ Convenience routine for rendering VAE based results.  By default returns results in HTML
    format, expecting to be part of a 3 column table body.  If not, returns results as a dictionary.
    If C1 is present, we attempt an interpolated rendering.
    """
    if C1 is not None:
        forward_res = model.forward_interpolate(X, C, C1)
    else:
        forward_res = model.forward(X, C)
    distances = evaluate_model(model, X, C, device=device)
    Y_h = forward_res["x"]
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
                save_model_epochs=50):
    """ Convenience routines for training VAEs defined here using unit type datasets.
    """
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
            forward_res = model.forward(X, C)
            Y_hat = reform_semantic_pixarray(forward_res["x"], N_DISTINCT_UNITS, OBS_WIDTH, OBS_HEIGHT)
            loss = 20 * multinomial_loss_fn(Y_hat, Y_gold)
            multi_loss = loss.item()

            # This is not likely correct, since we need to model the mutual information deltas, and these 
            # should not be against standard gaussians
            m1 = 0.2 * (gauss_KL_divg(forward_res['mu_w'], forward_res['logvar_w']) + \
                    gauss_KL_divg(forward_res['mu_z'], forward_res['logvar_z']))
            m1_loss = m1.item()


            #loss += m1  # See if M1 is causing us the issues here, seems the batch size was?
            #loss -= torch.sum(forward_res["x"])
            #loss -= torch.sum(forward_res["y_cond_z"])


            # TODO: Include log-likelihood P(X|W,Z) and P(Y)
            # TODO: Add mutual information minimization between H(Y) and H(Y|Z)
            # - May have to use
            train_loss = loss.item()
            epoch_loss += train_loss
            # tqdm_iter.set_description("Epoch {}, Multi-Loss={:.5f}\tM1={:.5f}\tLoss={:.5f}".format(epoch, multi_loss, m1_loss, epoch_loss))
            tqdm_iter.set_description("Epoch {}, Loss={:.5f}".format(epoch, epoch_loss))
            loss.backward()
            optimizer.step()
            steps += 1
        if save_model_dir and (epoch % save_model_epochs == 0):
            save_checkpoint("model_{}.pt".format(epoch), model, optimizer, save_model_dir, steps)
        with torch.no_grad():
            model.eval()
            val_loss_total = 0.
            for val_X, val_Y, val_C in test_dataloader:
                val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
                val_Y = torch.tensor(val_Y, dtype=torch.long).to(device)
                val_C = torch.tensor(val_C, dtype=torch.float32).to(device)
                forward_res = model.forward(val_X, val_C)
                Y_hat = reform_semantic_pixarray(forward_res["x"], N_DISTINCT_UNITS, OBS_WIDTH, OBS_HEIGHT)
                val_loss_total += multinomial_loss_fn(Y_hat, val_Y).item()
            print("Epoch {} Mean validation loss={}".format(epoch, val_loss_total / len(train_dataloader)))
            compute_divergence(model, test_dataloader, device=device)
            print("- - - - - - - - - -")
            model.train()
    if save_model_dir:
        save_checkpoint("model_{}.pt".format(epoch), model, optimizer, save_model_dir, steps)
    model.eval()


    
    
def make_encoder(N_INPUT_CHANNELS, N_HIDDEN):
    return nn.Sequential(
        nn.Conv2d(N_INPUT_CHANNELS, N_HIDDEN, kernel_size=4, stride=2),  # H = 31
        nn.ReLU(),
        nn.Conv2d(N_HIDDEN, N_HIDDEN * 2, kernel_size=4, stride=2),  # H = 14
        nn.ReLU(),
        nn.Conv2d(N_HIDDEN * 2, N_HIDDEN * 4, kernel_size=4, stride=2),  # H = 6
        nn.ReLU(),
        nn.Conv2d(N_HIDDEN * 4, N_HIDDEN * 8, kernel_size=4, stride=2),  # H = 2
        nn.ReLU(),
        nn.Flatten()
    )

class ConvCSVAE(nn.Module):
    def __init__(self, N_COND, N_TARGETS, N_LATENT=64, N_HIDDEN_CHANNELS=32,
                 N_HIDDEN_Z_C=16):
        super(ConvCSVAE, self).__init__()
        self.WIDTH = 64
        self.N_TARGETS = N_TARGETS
        self.N_COND, self.N_LATENT, self.N_HIDDEN = N_COND, N_LATENT, N_HIDDEN_CHANNELS


        self.encoder_W = make_encoder(N_TARGETS + N_COND, N_HIDDEN_CHANNELS)
        self.encoder_Z = make_encoder(N_TARGETS, N_HIDDEN_CHANNELS)

        self.h_W = N_HIDDEN_CHANNELS * 8 * 2 ** 2
        self.fc21_W = nn.Linear(self.h_W, N_LATENT)  # mu
        self.fc22_W = nn.Linear(self.h_W, N_LATENT)  # logvar

        self.h_Z = N_HIDDEN_CHANNELS * 8 * 2 ** 2
        self.fc21_Z = nn.Linear(self.h_Z, N_LATENT)  # mu
        self.fc22_Z = nn.Linear(self.h_Z, N_LATENT)  # logvar

        self.h = N_HIDDEN_CHANNELS * 8 * 2 ** 2
        self.fc3 = nn.Linear(2 * self.N_LATENT, self.h)  # Decode first layer

        self.decoder_X = nn.Sequential(
            Unflatten2D(self.h, 1, 1),
            nn.ConvTranspose2d(self.h, N_HIDDEN_CHANNELS * 4, kernel_size=5, stride=2),  # 5 x 5
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN_CHANNELS * 4, N_HIDDEN_CHANNELS * 2, kernel_size=5, stride=2),  # 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN_CHANNELS * 2, N_HIDDEN_CHANNELS, kernel_size=6, stride=2),  # 30 x 30
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN_CHANNELS, N_TARGETS, kernel_size=6, stride=2),  # 64 x 64
        )

        self.predict_C = nn.Sequential(
            nn.Linear(self.N_LATENT, N_HIDDEN_Z_C),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_Z_C, N_COND)
        )

    def is_conditional(self):
        return True

    def encode_W(self, x, C):
        # Take the input C and repeat it along each of the image input channels
        c1 = C.view(-1, C.shape[1], 1, 1).repeat(1, 1, self.WIDTH, self.WIDTH)
        x1 = torch.cat((x, c1), 1)
        h1 = self.encoder_W(x1)
        return self.fc21_W(h1), self.fc22_W(h1)

    def encode_Z(self, x):
        # Take the input C and repeat it along each of the image input channels
        h1 = self.encoder_Z(x)
        return self.fc21_Z(h1), self.fc22_Z(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z
        else:
            return mu

    def decode_X(self, z, w):
        z_combined = torch.cat((z, w), 1)
        h3 = self.fc3(z_combined)
        return self.decoder_X(h3)

    def forward(self, x, c):
        # Encode Z
        mu_z, logvar_z = self.encode_Z(x)
        z = self.reparameterize(mu_z, logvar_z)
        # Encode W
        mu_w, logvar_w = self.encode_W(x, c)
        w = self.reparameterize(mu_w, logvar_w)
        # Decode X and Y|Z
        x = self.decode_X(z, w)
        y_cond_z = self.predict_C(z)

        return {"x": x, "y_cond_z": y_cond_z,
                "mu_z": mu_z, "logvar_z": logvar_z,
                "mu_w":mu_w, "logvar_w": logvar_w }
    
    def forward_interpolate(self, x, c, c1, deterministic=False):
        # Encode Z
        mu_z, logvar_z = self.encode_Z(x)
        z = self.reparameterize(mu_z, logvar_z)
        # Encode W
        mu_w, logvar_w = self.encode_W(x, c1)
        w = self.reparameterize(mu_w, logvar_w)
        # Decode X and Y|Z
        x = self.decode_X(z, w)
        y_cond_z = self.predict_C(z)

        return {"x": x, "y_cond_z": y_cond_z,
                "mu_z": mu_z, "logvar_z": logvar_z,
                "mu_w":mu_w, "logvar_w": logvar_w }
    
    def forward_det(self, x, c):
        # Encode Z
        mu_z, logvar_z = self.encode_Z(x)
        z = mu_z
        # Encode W
        mu_w, logvar_w = self.encode_W(x, c)
        w = mu_w
        # Decode X and Y|Z
        x = self.decode_X(z, w)
        y_cond_z = self.predict_C(z)

        return {"x": x, "y_cond_z": y_cond_z,
                "mu_z": mu_z, "logvar_z": logvar_z,
                "mu_w":mu_w, "logvar_w": logvar_w }


class ConvCSVAEv2(nn.Module):
    """ ConvCSVAE where conditioning information is not repeated in each input channel, but concatenated to the
    convolutional encoding.  No Z, just W.
    """
    def __init__(self, N_COND, N_TARGETS, N_LATENT=64, N_HIDDEN_CHANNELS=32,
                 N_HIDDEN_Z_C=16):
        super(ConvCSVAEv2, self).__init__()
        self.WIDTH = 64
        self.N_TARGETS = N_TARGETS
        self.N_COND, self.N_LATENT, self.N_HIDDEN = N_COND, N_LATENT, N_HIDDEN_CHANNELS


        self.encoder_W = make_encoder(N_TARGETS, N_HIDDEN_CHANNELS)
        # Force everything through W
        # self.encoder_Z = make_encoder(N_TARGETS, N_HIDDEN_CHANNELS)


        self.h_W = (N_HIDDEN_CHANNELS * 8 * 2 ** 2) + N_COND
        self.fc21_W = nn.Linear(self.h_W, N_LATENT)  # mu
        self.fc22_W = nn.Linear(self.h_W, N_LATENT)  # logvar

        self.h_Z = N_HIDDEN_CHANNELS * 8 * 2 ** 2
        self.fc21_Z = nn.Linear(self.h_Z, N_LATENT)  # mu
        self.fc22_Z = nn.Linear(self.h_Z, N_LATENT)  # logvar

        self.h = N_HIDDEN_CHANNELS * 8 * 2 ** 2
        self.fc3 = nn.Linear(2 * self.N_LATENT, self.h)  # Decode first layer

        self.decoder_X = nn.Sequential(
            Unflatten2D(self.h, 1, 1),
            nn.ConvTranspose2d(self.h, N_HIDDEN_CHANNELS * 4, kernel_size=5, stride=2),  # 5 x 5
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN_CHANNELS * 4, N_HIDDEN_CHANNELS * 2, kernel_size=5, stride=2),  # 13 x 13
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN_CHANNELS * 2, N_HIDDEN_CHANNELS, kernel_size=6, stride=2),  # 30 x 30
            nn.ReLU(),
            nn.ConvTranspose2d(N_HIDDEN_CHANNELS, N_TARGETS, kernel_size=6, stride=2),  # 64 x 64
        )

        self.predict_C = nn.Sequential(
            nn.Linear(self.N_LATENT, N_HIDDEN_Z_C),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_Z_C, N_COND)
        )

    def is_conditional(self):
        return True

    def encode_W(self, x, C):
        h1_c = self.encoder_W(x)
        h1 = torch.cat((h1_c, C), 1)
        return self.fc21_W(h1), self.fc22_W(h1)

    #def encode_Z(self, x):
    #    # Take the input C and repeat it along each of the image input channels
    #    h1 = self.encoder_Z(x)
    #    return self.fc21_Z(h1), self.fc22_Z(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z
        else:
            return mu

    def decode_X(self, z, w):
        z_combined = torch.cat((z, w), 1)
        h3 = self.fc3(z_combined)
        return self.decoder_X(h3)

    def forward(self, x, c):
        # Encode W
        mu_w, logvar_w = self.encode_W(x, c)
        w = self.reparameterize(mu_w, logvar_w)
        # Decode X and Y|Z
        x = self.decode_X(w, w) # ADJUSTMENT: Force it all through W
        y_cond_z = self.predict_C(w)

        return {"x": x, "y_cond_z": y_cond_z,
                "mu_z": mu_z, "logvar_z": logvar_z,
                "mu_w":mu_w, "logvar_w": logvar_w }
    
    def forward_interpolate(self, x, c, c1, deterministic=False):
        # Encode Z
        #mu_z, logvar_z = self.encode_Z(x)
        #z = self.reparameterize(mu_z, logvar_z)
        # Encode W
        mu_w, logvar_w = self.encode_W(x, c1)
        w = self.reparameterize(mu_w, logvar_w)
        # Decode X and Y|Z
        x = self.decode_X(w, w) # ADJUSTMENT: Force it all through W
        y_cond_z = self.predict_C(w)

        return {"x": x, "y_cond_z": y_cond_z,
                "mu_z": mu_z, "logvar_z": logvar_z,
                "mu_w":mu_w, "logvar_w": logvar_w }
    
    def forward_det(self, x, c):
        # Encode Z
        # mu_z, logvar_z = self.encode_Z(x)
        # z = mu_z
        # Encode W
        mu_w, logvar_w = self.encode_W(x, c)
        w = mu_w
        # Decode X and Y|Z
        x = self.decode_X(w, w) 
        y_cond_z = self.predict_C(w)

        return {"x": x, "y_cond_z": y_cond_z,
                "mu_w":mu_w, "logvar_w": logvar_w }
