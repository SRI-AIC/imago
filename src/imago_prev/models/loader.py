import imago_prev.models.multinomial_vaes as mvaes
import imago_prev.models.csvae as csvaes
import imago_prev.models.csvae2 as csvaes2



def instance_model(model_name,metadata, N_HIDDEN, N_LATENT, device):
    if model_name=="convcondvae":
        model = mvaes.ConvCondVAE(N_TARGETS=metadata.N_DISTINCT_UNITS, N_COND=metadata.N_COND).to(device)
    elif model_name=="convcondvae_v2":
        model = mvaes.ConvCondVAEv2(N_TARGETS=metadata.N_DISTINCT_UNITS, N_COND=metadata.N_COND).to(device)
    elif model_name == "convvae":
        model = mvaes.ConvVAE(N_TARGETS=metadata.N_DISTINCT_UNITS,  N_HIDDEN=N_HIDDEN, N_LATENT=N_LATENT).to(device)
    elif model_name == "condvae":
        model = mvaes.CondVAE(metadata.N_DISTINCT_UNITS * metadata.OBS_WIDTH * metadata.OBS_HEIGHT,
                              N_COND=metadata.N_COND,
                          N_HIDDEN=N_HIDDEN, N_LATENT=N_LATENT).to(device)
    elif model_name == "vae":
        model = mvaes.VAE(metadata.N_DISTINCT_UNITS * metadata.OBS_WIDTH * metadata.OBS_HEIGHT,
                          N_HIDDEN=N_HIDDEN, N_LATENT=N_LATENT).to(device)
    elif model_name == "convcsvae":
        model = csvaes.ConvCSVAE(N_TARGETS=metadata.N_DISTINCT_UNITS, N_COND=metadata.N_COND).to(device)
    elif model_name == "convcsvae_v2":
        model = csvaes.ConvCSVAEv2(N_TARGETS=metadata.N_DISTINCT_UNITS, N_COND=metadata.N_COND).to(device)
    else:
        print("UNKNOWN MODEL TYPE!")
        return None
    return model