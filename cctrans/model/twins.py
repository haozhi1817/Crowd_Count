import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Block as TimmBlock

from model.regression_head import RegressionHead


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout=0.0
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupAttention(nn.Module):
    """
    Window Attention

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ws=1,
    ) -> None:
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(
            2, 3
        )  # (B, N, M, p, p, C)

        # (B, N, M, p, p, 3C) -> (B, NM, pp, 3, 8, C // 8) -> (3, B, NM, 8, pp, C // 8)
        qkv = (
            self.qkv(x)
            .reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, NM, 8, pp, pp)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (
            (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        )
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBlock(TimmBlock):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        drop=0,
        attn_drop=0,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(SBlock, self).__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )

    def forward(self, x, H, W):
        return super(SBlock, self).forward(x)


class GroupBlock(TimmBlock):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop=0,
        attn_drop=0,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        ws=1,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )
        del self.attn
        if ws == 1:
            self.attn = Attention(
                dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio
            )
        else:
            self.attn = GroupAttention(
                dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws
            )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim=768) -> None:
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        assert (
            img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), f"img_size {img_size} should be divided by patch_size {patch_size}"
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PVT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        block_cls=Block,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.embed_dim = embed_dims[-1]
        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(
                    PatchEmbed(img_size, patch_size, in_chans, embed_dims[i])
                )
            else:
                self.patch_embeds.append(
                    PatchEmbed(
                        img_size // patch_size // 2 ** (i - 1),
                        2,
                        embed_dims[i - 1],
                        embed_dims[i],
                    )
                )
            patch_num = (
                self.patch_embeds[-1].num_patches + 1
                if i == len(embed_dims) - 1
                else self.patch_embeds[-1].num_patches
            )
            self.pos_embeds.append(
                nn.Parameter(torch.zeros(1, patch_num, embed_dims[i]))
            )
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList(
                [
                    block_cls(
                        dim=embed_dims[k],
                        num_heads=num_heads[k],
                        mlp_ratio=mlp_ratios[k],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + i],
                        sr_ratio=sr_ratios[k],
                    )
                    for i in range(depths[k])
                ]
            )
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
        self.head = (
            nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        )

        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),
        )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class CPVTV2(PVT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        block_cls=Block,
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dims,
            num_heads,
            mlp_ratios,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            depths,
            sr_ratios,
            block_cls,
        )
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )
        self.regression = RegressionHead(embed_dims = embed_dims)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def no_weight_decay(self):
        return set(
            ["cls_token"]
            + ["pos_block." + n for n, p in self.pos_block.named_parameters()]
        )

    def forward_fatures(self, x):
        outputs = list()
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def forward(self, x):
        x = self.forward_fatures(x)
        mu = self.regression(x[1], x[2], x[3])
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


class PCPVT(CPVTV2):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        depths=[4, 4, 4],
        sr_ratios=[4, 2, 1],
        block_cls=SBlock,
    ) -> None:
        super(PCPVT, self).__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dims,
            num_heads,
            mlp_ratios,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            depths,
            sr_ratios,
            block_cls,
        )


class ALTGVT(PCPVT):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        depths=[4, 4, 4],
        sr_ratios=[4, 2, 1],
        block_cls=GroupBlock,
        wss=[7, 7, 7],
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dims,
            num_heads,
            mlp_ratios,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            depths,
            sr_ratios,
            block_cls,
        )

        del self.blocks
        self.wss = wss
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList(
                [
                    block_cls(
                        dim=embed_dims[k],
                        num_heads=num_heads[k],
                        mlp_ratio=mlp_ratios[k],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + i],
                        sr_ratio=sr_ratios[k],
                        ws=1 if i % 2 == 1 else wss[k],
                    )
                    for i in range(depths[k])
                ]
            )
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)


def alt_gvt_small():
    model = ALTGVT(
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 10, 4],
        wss=[8, 8, 8, 8],
        sr_ratios=[8, 4, 2, 1],
    )
    return model


def alt_gvt_large():
    model = ALTGVT(
        patch_size=4,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        depths=[2, 2, 18, 2],
        wss=[8, 8, 8, 8],
        sr_ratios=[8, 4, 2, 1],
    )
    return model


if __name__ == "__main__":
    img = torch.zeros(size=(4, 3, 256, 256))
    model = alt_gvt_small()
    mu, mu_norm = model(img)
    print(mu.shape, mu_norm.shape)
