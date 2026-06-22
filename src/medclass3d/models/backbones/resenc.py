from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist

from medclass3d.training.trainer import BaseModel
from medclass3d.models.heads.classification import ClassificationHead, ClassificationHead_MLP


def get_first_valid_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the specified keys found: {keys}")



class ResEncoder(Module):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoder, self).__init__()

        self.res_unet = ResidualEncoderUNet(
            hypparams["input_channels"],
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            num_classes=hypparams["num_classes"],
        )
        self.res_unet.encoder.return_skips = False

        if hypparams["pretrained"]:
            self.res_unet = load_pretrained_weights(
                self.res_unet,
                hypparams["chpt_path"],
            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.res_unet.named_parameters():
                    param.requires_grad = False

    def forward(self, x):

        x = self.res_unet.encoder(x).mean(dim=[2, 3, 4])

        return x


class ResEncoder_Classifier(BaseModel):
    """ResEncoder backbone with a linear classification head.

    Use with ``task: 'Classification'``. Output shape is ``[B, num_classes]``
    logits; the trainer applies softmax (multiclass) or sigmoid (multilabel)
    before computing metrics.
    """

    def __init__(self, **hypparams):
        super().__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        self.cls_head = ClassificationHead(
            embed_dim=320,
            num_classes=hypparams["num_classes"],
            dropout=hypparams.get("classification_head_dropout", 0.1),
            patch_aggregation_method=hypparams.get("token_aggregation_method", None),
            cls_token_available=False,
        )

        # Optionally restore cls_head weights from a checkpoint that was saved
        # with the same head shape.
        if hypparams.get("pretrained", False):
            ckpt = torch.load(hypparams["chpt_path"], map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            for name, param in state_dict.items():
                if name.startswith("cls_head") and name in self.state_dict():
                    if self.state_dict()[name].shape == param.shape:
                        self.state_dict()[name].copy_(param)

    def forward(self, x):
        x = self.encoder(x)
        return self.cls_head(x)


class ResEncoder_Classifier_MLP(BaseModel):
    """MLP-head variant of :class:`ResEncoder_Classifier`."""

    def __init__(self, **hypparams):
        super().__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        self.cls_head = ClassificationHead_MLP(
            embed_dim=320,
            num_classes=hypparams["num_classes"],
            dropout=hypparams.get("classification_head_dropout", 0.1),
            patch_aggregation_method=hypparams.get("token_aggregation_method", None),
            cls_token_available=False,
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.cls_head(x)


def load_pretrained_weights(
    resenc_model,
    pretrained_weights_file,
):
    if dist.is_initialized():
        saved_model = torch.load(
            pretrained_weights_file,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(pretrained_weights_file, weights_only=False)
    if 'network_weights' in saved_model:
        pretrained_dict = saved_model['network_weights']
    elif 'state_dict' in saved_model:
        pretrained_dict = saved_model['state_dict']
    else:
        raise KeyError("No compatible weight dictionary ('network_weights' or 'state_dict') found in checkpoint")


    if isinstance(resenc_model, DDP):
        mod = resenc_model.module
    else:
        mod = resenc_model
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    in_conv_weights_model = get_first_valid_key(model_dict, [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
    ])

    in_conv_weights_pretrained = get_first_valid_key(pretrained_dict, [
        "encoder.stem.convs.0.all_modules.0.weight",
        "encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
    ])


    in_channels_model = in_conv_weights_model.shape[1]
    in_channels_pretrained = in_conv_weights_pretrained.shape[1]

    if in_channels_model != in_channels_pretrained:
        assert in_channels_pretrained == 1, (
            f"The input channels do not match. Pretrained model: {in_channels_pretrained}; your network: "
            f"your network: {in_channels_model}"
        )

        # Inflate the input stem from 1 -> in_channels_model channels by repeating
        # the single pretrained channel and rescaling so the summed response is
        # preserved at init. We match the stem conv weights *by shape* rather than
        # by shared storage (data_ptr): the latter is fragile and only catches the
        # one key that aliases the sampled tensor. Checkpoints like S3D store
        # 'encoder.stem.convs.0.conv.weight' and '...all_modules.0.weight' as
        # distinct tensors, so the data_ptr approach left 'conv.weight' at 1
        # channel and failed the shape check below. A stem input conv is uniquely
        # identifiable: the only weight whose in-channel dim equals
        # in_channels_pretrained in the checkpoint while the matching model key
        # expects in_channels_model.
        inflated_keys = []
        for key, w in pretrained_dict.items():
            if key not in model_dict:
                continue
            mw = model_dict[key]
            if (
                w.ndim == mw.ndim
                and w.ndim >= 2
                and w.shape[0] == mw.shape[0]
                and w.shape[1] == in_channels_pretrained
                and mw.shape[1] == in_channels_model
                and w.shape[2:] == mw.shape[2:]
            ):
                repeat_dims = [1, in_channels_model // in_channels_pretrained] + [1] * (w.ndim - 2)
                pretrained_dict[key] = w.repeat(*repeat_dims) / in_channels_model
                inflated_keys.append(key)

        assert inflated_keys, (
            "Input channels differ but no stem conv weight could be inflated. "
            f"Expected a checkpoint weight with in-channel dim {in_channels_pretrained} "
            f"matching a model weight with in-channel dim {in_channels_model}."
        )
        print(
            f"Your network has {in_channels_model} input channels. The single input channel "
            f"of the pretrained model is repeated {in_channels_model} times (rescaled by "
            f"1/{in_channels_model}) for: {inflated_keys}"
        )

    skip_strings_in_pretrained = [".seg_layers."]
    skip_strings_in_pretrained.extend(["decoder.stages", "decoder.transpconvs"])

    final_pretrained_dict = {}
    for key, v in pretrained_dict.items():
        if key in model_dict and all(
            [i not in key for i in skip_strings_in_pretrained]
        ):
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model "
                f"does not seem to be compatible with your network."
            )
            final_pretrained_dict[key] = v

    model_dict.update(final_pretrained_dict)

    mod.load_state_dict(model_dict)

    return mod