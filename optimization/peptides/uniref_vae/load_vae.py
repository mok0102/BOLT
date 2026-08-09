import torch

from uniref_vae.data import DataModuleKmers
from uniref_vae.data import collate_fn
from uniref_vae.transformer_vae_unbounded import InfoTransformerVAE as UnirefVAE

ENCODER_DIM = 256
DECODER_DIM = 256
KL_FACTOR = 0.0001
ENCODER_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6


def load_vae(
    path_to_vae_statedict: str,
    dim: int = 256,
    max_string_length=150,
):
    return load_uniref_vae(
        path_to_vae_statedict,
        dim=dim,
        max_string_length=max_string_length,
    )


# example function to load vae, loads uniref vae
def load_uniref_vae(
    path_to_vae_statedict,
    dim=256,
    max_string_length=150,
):
    data_module = DataModuleKmers(
        batch_size=10,
        k=1,
        load_data=False,
    )
    dataobj = data_module.train
    vae = UnirefVAE(
        dataset=dataobj,
        d_model=dim // 2,
        kl_factor=KL_FACTOR,
        encoder_dim_feedforward=ENCODER_DIM,
        decoder_dim_feedforward=DECODER_DIM,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        decoder_num_layers=DECODER_NUM_LAYERS,
    )

    # load in state dict of trained model:
    if path_to_vae_statedict:
        state_dict = torch.load(path_to_vae_statedict)
        vae.load_state_dict(state_dict, strict=True)
    vae = vae.cuda()
    vae = vae.eval()

    # set max string length that VAE can generate
    vae.max_string_length = max_string_length

    return vae, dataobj


def vae_forward(xs_batch, dataobj, vae):
    """Input:
        a list xs
    Output:
        z: tensor of resultant latent space codes
            obtained by passing the xs through the encoder
        vae_loss: the total loss of a full forward pass
            of the batch of xs through the vae
            (ie reconstruction error)
    """
    # assumes xs_batch is a batch of smiles strings
    tokenized_seqs = dataobj.tokenize_sequence(xs_batch)
    encoded_seqs = [dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
    X = collate_fn(encoded_seqs)
    dict = vae(X.cuda())
    vae_loss, z = dict["loss"], dict["z"]
    z = z.reshape(-1, 256)

    return z, vae_loss


def vae_decode(z, vae, dataobj):
    """Input
        z: a tensor latent space points (bsz, self.dim)
    Output
        a corresponding list of the decoded input space
        items output by vae decoder
    """
    z = z.cuda()
    vae = vae.eval()
    vae = vae.cuda()
    # sample molecular string form VAE decoder
    sample = vae.sample(z=z.reshape(-1, 2, 128))
    # grab decoded aa strings
    decoded_seqs = [dataobj.decode(sample[i]) for i in range(sample.size(-2))]

    return decoded_seqs
