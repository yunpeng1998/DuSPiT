# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import SubPatchVisionRotaryEmbeddingFast, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm



import torch
import math

def get_2d_sincos_pos_embed_l(H, W, dim, device="cpu"):
    """
    return: (H*W, dim)

    """
    assert dim % 4 == 0, "dim must be divisible by 4"


    grid_y = torch.arange(H, device=device)
    grid_x = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)


    dim_half = dim // 2
    dim_quarter = dim_half // 2

    omega = torch.arange(dim_quarter, device=device)
    omega = 1. / (10000 ** (omega / dim_quarter))

    out_y = torch.einsum("n,d->nd", grid_y, omega)
    out_x = torch.einsum("n,d->nd", grid_x, omega)

    pos_emb = torch.cat([
        torch.sin(out_y), torch.cos(out_y),
        torch.sin(out_x), torch.cos(out_x)
    ], dim=1)

    return pos_emb  # (H*W, dim)



def modulate(x, shift, scale):
    """
    x:     (Bx, L, D)
    shift: (Bs, D) or (Bs, L, D)
    scale: (Bs, D) or (Bs, L, D)

    """
    assert x.dim() == 3, f"x must be (B,L,D), got {x.shape}"
    Bx, L, D = x.shape

    assert shift.dim() in (2, 3), f"shift must be (B,D) or (B,L,D), got {shift.shape}"
    assert scale.dim() in (2, 3), f"scale must be (B,D) or (B,L,D), got {scale.shape}"

    # ---- normalize shift/scale shapes & check dims ----
    if shift.dim() == 2:
        Bs, Ds = shift.shape
        assert Ds == D, f"dim mismatch: x={x.shape}, shift={shift.shape}"
    else:
        Bs, Ls, Ds = shift.shape
        assert Ls == L and Ds == D, f"dim mismatch: x={x.shape}, shift={shift.shape}"

    if scale.dim() == 2:
        Bs2, Ds2 = scale.shape
        assert Ds2 == D, f"dim mismatch: x={x.shape}, scale={scale.shape}"
    else:
        Bs2, Ls2, Ds2 = scale.shape
        assert Ls2 == L and Ds2 == D, f"dim mismatch: x={x.shape}, scale={scale.shape}"

    assert Bs == Bs2, f"batch mismatch: shift={shift.shape}, scale={scale.shape}"

    # ---- batch align (repeat cond if needed) ----
    if Bs != Bx:
        assert Bx % Bs == 0, f"Batch not multiple: x={x.shape}, shift={shift.shape}, scale={scale.shape}"
        repeat = Bx // Bs

        # repeat along batch dim
        repeat = int(repeat)


        shift = shift.repeat_interleave(repeat, dim=0)
        scale = scale.repeat_interleave(repeat, dim=0)

    # ---- expand (B,D) -> (B,L,D) if needed ----
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)  # (Bx,1,D)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)  # (Bx,1,D)

    return x * (1 + scale) + shift





class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x





class CrossAttention(nn.Module):
    def __init__(self, dim, max_tokens, num_heads=8,  qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = 64
        patch_dim = 64
        self.patch_dim = patch_dim
          
        self.max_tokens = max_tokens

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q = nn.Linear(patch_dim, head_dim , bias=qkv_bias)
        self.kv = nn.Linear(patch_dim, head_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_drop = nn.Dropout(proj_drop)
        # batched per-token projection params (等价于 ModuleList([Linear]*64))
        self.W = nn.Parameter(torch.empty(self.max_tokens, self.patch_dim, head_dim))
        self.b = nn.Parameter(torch.empty(self.max_tokens, self.patch_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)
    def forward(self, x1, x2, rope1, rope2):


        B, N2, C2 = x2.shape
        _, N1, C = x1.shape

        q = self.q(x2).reshape(B, N2, 1, 1, C // 1).permute(2, 0, 3, 1, 4)[0]
        kv1 = self.kv(x1).reshape(B, N1, 2, 1, C // 1).permute(2, 0, 3, 1, 4)



        k1, v1 = kv1[0], kv1[1]   # make torchscript happy (cannot use tensor as tuple)

        k1 = self.k_norm(k1)
        q = self.q_norm(q)

        
        k1 = rope1(k1)
        q = rope2(q)

        # x = scaled_dot_product_attention(q, k1, v1, dropout_p=self.attn_drop.p if self.training else 0.)
        x = F.scaled_dot_product_attention(
            q, k1, v1,
            attn_mask=None,      
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )
        x = x.transpose(1, 2).reshape(B, N2, C)

        
        # === fast batched per-token linear ===
        W = self.W#[:N2]  # [N2, patch_dim, head_dim]
        b = self.b#[:N2]  # [N2, patch_dim]
        out = torch.einsum('bnh,nph->bnp', x, W) + b  # [B, N2, patch_dim]
        out = self.proj_drop(out)
        return out






class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))



class FinalLayer(nn.Module):
    """
    The final layer of FiT.
    """
    def __init__(self, hidden_size, mlp_hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    #@torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class FiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, 6 * hidden_size, bias=True)
        )

    #@torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if x.shape[0] != gate_msa.shape[0]:
            repeat  = x.shape[0] // gate_msa.shape[0]
            gate_msa = gate_msa.repeat_interleave(repeat, dim=0)
            gate_mlp = gate_mlp.repeat_interleave(repeat, dim=0)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x





class CrossFiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, multi_times, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = CrossAttention(mlp_hidden_size, num_heads=num_heads, max_tokens=multi_times**2, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        cood = get_2d_sincos_pos_embed_l(multi_times, multi_times, 1024)
        self.register_buffer("cood", cood, persistent=False)

        self.mlp = TokenWiseSwiGLUFFN(hidden_size, mlp_hidden_dim, num_tokens=multi_times**2, drop=proj_drop, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    #@torch.compile
    def forward(self, x1, x2,  c, feat_rope1=None, feat_rope2=None):

        c = c.unsqueeze(1).expand(-1, self.cood.shape[0], -1) + self.cood

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        if x2.shape[0] != gate_msa.shape[0]:
            repeat  = x2.shape[0] // gate_msa.shape[0]
            gate_msa = gate_msa.repeat_interleave(int(repeat), dim=0)
            gate_mlp = gate_mlp.repeat_interleave(int(repeat), dim=0)

        x2 = x2 + gate_msa * self.attn(self.norm1(x1), modulate(self.norm1(x2), shift_msa, scale_msa), rope1=feat_rope1, rope2=feat_rope2)
        x2 = x2 + gate_mlp * self.mlp( modulate(self.norm2(x2), shift_mlp, scale_mlp)) #+ gate_x1 * x1_plus
        return x2


class TokenWiseSwiGLUFFN(nn.Module):
    """
    x: [B, N, dim]
    每个 token i 有自己的一套参数:
      w12[i]: [2*h_i, dim]
      w3[i] : [dim, h_i]
    其中 h_i = int((hidden_dim * 2/3))
    """
    def __init__(self, dim: int, hidden_dim: int, num_tokens: int = 64, drop: float = 0.0, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        h = int(hidden_dim * 2 / 3)
        self.h = h
        self.drop = drop

        # w12: [N, 2h, dim], b12: [N, 2h]
        self.w12 = nn.Parameter(torch.empty(num_tokens, 2 * h, dim))
        self.b12 = nn.Parameter(torch.zeros(num_tokens, 2 * h)) if bias else None

        # w3: [N, dim, h], b3: [N, dim]
        self.w3 = nn.Parameter(torch.empty(num_tokens, dim, h))
        self.b3 = nn.Parameter(torch.zeros(num_tokens, dim)) if bias else None

        nn.init.xavier_uniform_(self.w12)
        nn.init.xavier_uniform_(self.w3)

        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, dim]
        return: [B, N, dim]
        """
        B, N, D = x.shape

        w12 = self.w12[:N]
        x12 = torch.einsum("bnd,ntd->bnt", x, w12)
        if self.b12 is not None:
            x12 = x12 + self.b12[:N][None, :, :]


        x1, x2 = x12.chunk(2, dim=-1)         # [B,N,h] each
        hidden = F.silu(x1) * x2              # [B,N,h]
        hidden = self.ffn_dropout(hidden)


        # out: [B, N, dim]
        w3 = self.w3[:N]                      # [N, dim, h]
        out = torch.einsum("bnh,ndh->bnd", hidden, w3)
        if self.b3 is not None:
            out = out + self.b3[:N][None, :, :]

        return out




class FiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        hidden_size2=1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        dim_times = hidden_size2 // hidden_size #

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        self.x_embedder2 = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size2, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size


        self.sub_feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.sub_feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )


        base = hidden_size2 // int(half_head_dim * 2)
        self.base = base
        self.multi_times = int(math.sqrt(base))
        self.sub_feat_rope_2 = SubPatchVisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
            multi_times = int(math.sqrt(base))
        )

        # transformer
        self.blocks = nn.ModuleList([
            FiTBlock(hidden_size, num_heads, hidden_size, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        self.blocks_2 = nn.ModuleList([
            CrossFiTBlock(int(hidden_size2 // base), num_heads, hidden_size, self.multi_times, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size2, hidden_size, patch_size, self.out_channels)



        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        w12 = self.x_embedder2.proj1.weight.data
        nn.init.xavier_uniform_(w12.view([w12.shape[0], -1]))
        w22 = self.x_embedder2.proj2.weight.data
        nn.init.xavier_uniform_(w22.view([w22.shape[0], -1]))
        nn.init.constant_(self.x_embedder2.proj2.bias, 0)
        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # for block in self.blocks_2:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb


        x_raw = x

        x = self.x_embedder(x_raw)

        x += self.pos_embed
        x2 = self.x_embedder2(x_raw)


        B, L, hidden_size = x.shape
        _, _, hidden_size2 = x2.shape
        dim_times =  hidden_size2 // hidden_size

        x2 = x2.view(B, L, self.base, int(hidden_size2//self.base))


        x2 = x2.reshape(B * L, self.base, int(hidden_size2//self.base))

        for i, block in enumerate(self.blocks):

            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb

                x = torch.cat([in_context_tokens, x], dim=1)


            x = block(x, c, self.sub_feat_rope if i < self.in_context_start else self.sub_feat_rope_incontext)

            if self.in_context_len > 0 and i >= self.in_context_start:
                x_l = x[:, self.in_context_len:]
            else:
                x_l = x


            x_l = x_l.view(B, L, 16, int(hidden_size//16))

            x_l = x_l.reshape(B * L, 16, int(hidden_size//16))

            x2 = self.blocks_2[i](x_l, x2, c, self.sub_feat_rope, self.sub_feat_rope_2)

        
        x = x[:, self.in_context_len:]

        x2 = x2.reshape(B, L, self.base, hidden_size2 // self.base)
        x2 = x2.reshape(B, L, hidden_size2)


        x2 = self.final_layer(x2, c)

        output = self.unpatchify(x2, self.patch_size)

        return output




