"""
Constructs the SVM around features
"""

from absl import logging

import sklearn.utils as skutils
from sklearn.svm import LinearSVC

import sklearn

logging.set_verbosity(logging.INFO)

from absl import logging

logging.set_verbosity(logging.INFO)

from imago.models.model import PVaeModel
import os
from tqdm import tqdm

from imago_prev import IMAGO_ROOT
from torch.utils.data import DataLoader

from imago.utils import *
from imago.model_utils import load_checkpoint

from imago.domains.pysc2 import mcbox_spec, latent2_target_specs, load_y1_as_trajs
from imago.domains.pysc2.pysc2_featurizer import IDPFeaturizer

# # Setup for data labeling
# 
# - Take incremental steps along each direction
# - Record trajectory as a numpy tensor in original form
# - Record adjacent Z tensor marking location
# - Record at which step things went bad
# 
# ## TODO
# - Convert start_res observation back into a (3, 64, 64) individual integer tensor (same as observations stored in IDPDataset)
#   - Rationale: Easier to work with than individual dictionaries (for people not neck deep in SC2)


DEBUG = False
DEVICE="cuda:0"

#W_SCALING_FACTOR = 20
SPLIT_AT = 0.8
W_SCALING_FACTOR = 2

DEBUG_STR = ""
#if DEBUG:
#    DEBUG_STR = "DEBUG"

CAML_ROOT = os.path.join(IMAGO_ROOT, "..")
EXP_NAME = "caml_y1_trajs_{}_splitat={}".format(DEBUG_STR, SPLIT_AT)
MODEL_ROOT = os.path.join(CAML_ROOT, "output/pvae/demos/{}".format(EXP_NAME))
MODEL_FPATH = os.path.join(MODEL_ROOT, "model.pt")

DEVICE='cuda:0'

model = PVaeModel(mcbox_spec,
                  latent2target_specs=latent2_target_specs,
                  device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
load_checkpoint(MODEL_FPATH, model, optimizer, device=DEVICE)

train_dataset, test_dataset, train_trajs, test_trajs = load_y1_as_trajs(debug=DEBUG, split_at=SPLIT_AT)

# Avoid shuffling, so ordering is consistent.  Shuffling will be performed
# when the SVM is trained
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

FEATURIZER_FPATH = os.path.join(MODEL_ROOT, "featurizer.json")
idp_featurizer = IDPFeaturizer(train_dataloader, mcbox_spec)

train_feats = idp_featurizer.featurize(train_dataloader)
test_feats = idp_featurizer.featurize(test_dataloader)
idp_featurizer.save(FEATURIZER_FPATH)

# Construct the latents
train_Zs = []
for datum in tqdm(train_dataloader):
    O = datum[0]
    Ohat, Z_mu, Z_logvar, Z, That = model(O)
    train_Zs.extend(ensure_numpy(Z))
train_Zs = np.stack(train_Zs)


# Go through each of the features and generate the SVM
feat_planes = {}
for feat_label in idp_featurizer.all_seen_feats:
    pos_locations = train_feats[feat_label] == 1
    pos_offsets = np.arange(len(train_feats))[pos_locations.values]
    train_Y = np.zeros(len(train_Zs))
    train_Y[pos_offsets] = 1
    train_Zs1, train_Y1 = skutils.shuffle(train_Zs, train_Y, random_state=42)
    clf = LinearSVC(random_state=1337, tol=1e-5, max_iter=1e6)
    clf.fit(train_Zs1, train_Y1)
    W = clf.coef_
    W = W.reshape(-1)
    feat_planes[feat_label] = W
    print("- - - - - - - - -\nTraining performance, pos_label={}".format(feat_label))
    print(sklearn.metrics.classification_report(train_Y, clf.predict(train_Zs)))

W_FPATH = os.path.join(MODEL_ROOT, "caml_y1_Ws.npz")
    
np.savez(W_FPATH, **feat_planes)
W2 = np.load(W_FPATH)
