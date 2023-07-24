import pdb
from absl import logging
import torch
import json
from tqdm import tqdm
import pandas as pd

from imago.domains.pysc2 import mcbox_spec
from imago.utils import scan_stats, ensure_numpy

"""
Implements a Featurizer that mimics the main Replay based one, but uses the
Interestingness Data Point (IDP) as inputs.
"""

RED = 'red'
BLUE = 'blue'
ALLEGIANCES = [RED, BLUE]

class IDPFeaturizer(object):
    def __init__(self, train_dataloader, mcbox_spec):
        """ Sets up the featurizer using the statistics from the
        given training set.
        """
        red_counts = []
        blue_counts = []
        self.all_seen_feats = set()
        self.mcbox_spec = mcbox_spec

        if isinstance(train_dataloader, str):
            # Open the JSON file and populate stats directly
            logging.info("Loading featurizer config from {}".format(train_dataloader))
            with open(train_dataloader, 'r') as f:
                featurizer_spec = json.load(f)
                self.all_seen_feats = set(featurizer_spec['seen_feats'])
                self.red_mu = featurizer_spec['red_mu']
                self.red_std = featurizer_spec['red_std']
                self.blue_mu = featurizer_spec['blue_mu']
                self.blue_std = featurizer_spec['blue_std']
        else:
            # Load from Torch Dataloader
            logging.info("Inferring suff stats from dataloader, size={}".format(len(train_dataloader)))
            for datum in tqdm(train_dataloader):
                O = datum[0]
                red_counts.extend(self._count_red(O))
                blue_counts.extend(self._count_blue(O))
            self.red_mu, self.red_std, _, _ = scan_stats(red_counts)
            self.blue_mu, self.blue_std, _, _ = scan_stats(blue_counts)

    def __str__(self):
        return "IDPFeaturizer, red mu/std={:.5f}/{:.5f}, blue mu/std={:.5f}/{:.5f}".format(
            self.red_mu, self.red_std,
            self.blue_mu, self.blue_std
            )

    def featurize(self, dataloader, default_na=0):
        """ Featurizes each of the instances in the given batch, returning
        a Pandas Dataframe
        """
        all_inst_feats = []
        all_feat_labels = set()
        for O in dataloader:
            if isinstance(O, list):
                O = O[0]  # Original datum passed in, get the observation
            red_labels = self._get_size_labels(RED, self._count_red(O))
            blue_labels = self._get_size_labels(BLUE, self._count_blue(O))
            for idx in range(len(O)):
                inst_feats = {}
                inst_feats[red_labels[idx]] = 1
                inst_feats[blue_labels[idx]] = 1
                all_feat_labels.update(inst_feats.keys())
                self.all_seen_feats.update(inst_feats.keys())
                all_inst_feats.append(inst_feats)

        # Now go through all known features and construct a dataframe
        updated = []
        for idx, inst_feats in enumerate(all_inst_feats):
            for feat_label in all_feat_labels:
                if feat_label not in inst_feats:
                    inst_feats[feat_label] = default_na
            updated.append(inst_feats)
        return pd.DataFrame(updated)

    def _get_size_labels(self, allegiance, counts):
        labels = []
        for count in counts:
            label = 'med'
            if allegiance == 'red':
                if count <= (self.red_mu - self.red_std):
                    label = "small"
                elif count >= (self.red_mu + self.red_std):
                    label = "large"
            elif allegiance == 'blue':
                if count <= (self.blue_mu - self.blue_std):
                    label = "small"
                elif count >= (self.blue_mu + self.blue_std):
                    label = "large"
            labels.append("{}_{}".format(allegiance, label))
        return labels

    def _count_red(self, O):
        R = self.mcbox_spec.get_channel_slice(O, 'player_relative')
        return ensure_numpy(torch.sum(R == 4, (1,2)))

    def _count_blue(self, O):
        R = self.mcbox_spec.get_channel_slice(O, 'player_relative')
        return ensure_numpy(torch.sum(R == 1, (1,2)))

    def save(self, fpath):
        info = {
            "red_mu": self.red_mu,
            "red_std": self.red_std,
            "blue_mu": self.blue_mu,
            "blue_std": self.blue_std,
            "seen_feats": list(self.all_seen_feats)
            }
        with open(fpath, 'w') as f:
            json.dump(info, f, indent=2)
