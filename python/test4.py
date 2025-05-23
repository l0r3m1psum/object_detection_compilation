import dataclasses
import enum
import os
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import tvm
from tvm import dlight, relax, te, tir
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache
from tvm.runtime import ShapeTuple

@dataclasses.dataclass
class LlamaConfig:
	hidden_size: int = 2048
	intermediate_size: int = 5632
	num_attention_heads: int = 32
	num_hidden_layers: int = 22
	rms_norm_eps: float = 1e-05
	vocab_size: int = 32000
	rope_theta: int = 10000
	context_window_size: int = 2048
	prefill_chunk_size: int = 2048
	num_key_value_heads: int = 4
	head_dim: int = 64  # hidden_size // num_attention_heads

dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)

class RopeMode(enum.IntEnum):
	"""The RoPE mode of the Paged KV cache.
	If it is none, the KV cache will not apply RoPE to q and k.
	If it is normal, RoPE will be applied to k before adding k to cache.
	Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
	"""

	NONE = 0
	NORMAL = 1
	INLINE = 2

# Gate and up projection are fused
class LlamaFFN(nn.Module):
	def __init__(self, config: LlamaConfig):
		super().__init__()
		self.gate_up_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=2 * config.intermediate_size,
			bias=False,
		)
		self.down_proj = nn.Linear(
			config.intermediate_size,
			config.hidden_size,
			bias=False
		)

	def forward(self, x: Tensor):
		concat_x1_x2 = self.gate_up_proj(x)
		x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
		return self.down_proj(op.silu(x1) * x2)

class LlamaAttention(nn.Module):
	def __init__(self, config: LlamaConfig):
		self.head_dim = config.head_dim
		self.num_q_heads = config.num_attention_heads
		self.num_kv_heads = config.num_key_value_heads

		# horizontal fusion on QKV projection
		self.qkv_proj = nn.Linear(
			in_features=config.hidden_size,
			out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
			bias=False,
		)
		self.o_proj = nn.Linear(
			self.num_q_heads * self.head_dim,
			config.hidden_size,
			bias=False
		)

	def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
		d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
		b, s, _ = hidden_states.shape
		# QKV Projection
		qkv = self.qkv_proj(hidden_states)
		qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
		# Attention
		output = op.reshape(
			paged_kv_cache.attention_with_fused_qkv(
				layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5
			),
			(b, s, h_q * d),
		)
		# Output Projection
		return self.o_proj(output)

class LlamaDecoderLayer(nn.Module):
	def __init__(self, config: LlamaConfig):
		rms_norm_eps = config.rms_norm_eps
		self.self_attn = LlamaAttention(config)
		self.mlp = LlamaFFN(config)
		self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
		self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

	def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
		hidden_states += self.self_attn(
			self.input_layernorm(hidden_states), paged_kv_cache, layer_id
		)
		hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
		return hidden_states

class LlamaModel(nn.Module):
	def __init__(self, config: LlamaConfig):
		assert config.hidden_size % config.num_attention_heads == 0
		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
		self.layers = nn.ModuleList(
			[LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
		)
		self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

	def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
		hidden_states = input_embed
		for layer_id, layer in enumerate(self.layers):
			hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
		hidden_states = self.norm(hidden_states)
		return hidden_states

class LlamaForCasualLM(nn.Module):
	def __init__(self, config: LlamaConfig):
		self.model = LlamaModel(config)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		self.num_hidden_layers = config.num_hidden_layers
		self.num_attention_heads = config.num_attention_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.head_dim = config.head_dim
		self.hidden_size = config.hidden_size
		self.vocab_size = config.vocab_size
		self.rope_theta = config.rope_theta
		self.dtype = "float32"

	def to(self, dtype: Optional[str] = None):
		super().to(dtype=dtype)
		if dtype is not None:
			self.dtype = dtype

	def embed(self, input_ids: Tensor):
		return self.model.embed_tokens(input_ids)

	def get_logits(self, hidden_states: Tensor):
		logits = self.lm_head(hidden_states)
		if logits.dtype != "float32":
			logits = logits.astype("float32")
		return logits

	def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
		def _index(x: te.Tensor):  # x[:-1,:]
			b, s, d = x.shape
			return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

		hidden_states = self.model(input_embed, paged_kv_cache)
		hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
		logits = self.get_logits(hidden_states)
		return logits, paged_kv_cache

	def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
		hidden_states = self.model(input_embed, paged_kv_cache)
		logits = self.get_logits(hidden_states)
		return logits, paged_kv_cache

	def create_tir_paged_kv_cache(
		self,
		max_batch_size: tir.Var,
		max_total_seq_len: tir.Var,
		prefill_chunk_size: tir.Var,
		page_size: tir.Var,
	) -> PagedKVCache:
		return TIRPagedKVCache(
			attn_kind="mha",
			max_batch_size=max_batch_size,
			max_total_seq_len=max_total_seq_len,
			prefill_chunk_size=prefill_chunk_size,
			page_size=page_size,
			support_sliding_window=0,
			layer_partition=relax.ShapeExpr([0, self.num_hidden_layers]),
			num_hidden_layers=self.num_hidden_layers,
			num_attention_heads=self.num_attention_heads,
			num_key_value_heads=self.num_key_value_heads,
			qk_head_dim=self.head_dim,
			v_head_dim=self.head_dim,
			mla_original_qk_head_dim=0,
			mla_original_v_head_dim=0,
			rope_mode=RopeMode.NORMAL,
			rope_scale=1,
			rope_theta=self.rope_theta,
			rope_scaling={},
			rope_ext_factors=relax.PrimValue(0),
			rotary_dim=self.head_dim,
			dtype=self.dtype,
			target=target,
			enable_disaggregation=False,
		)

	def get_default_spec(self):
		mod_spec = {
			"embed": {
				"input_ids": nn.spec.Tensor(["seq_len"], "int32"),
				"$": {
					"param_mode": "packed",
					"effect_mode": "none",
				},
			},
			"prefill": {
				"input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
				"paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
				"$": {
					"param_mode": "packed",
					"effect_mode": "none",
				},
			},
			"decode": {
				"input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
				"paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
				"$": {
					"param_mode": "packed",
					"effect_mode": "none",
				},
			},
			"create_tir_paged_kv_cache": {
				"max_batch_size": int,
				"max_total_seq_len": int,
				"prefill_chunk_size": int,
				"page_size": int,
				"$": {
					"param_mode": "none",
					"effect_mode": "none",
				},
			},
		}
		return nn.spec.ModuleSpec.from_raw(mod_spec, self)

model_config = LlamaConfig()
model = LlamaForCasualLM(model_config)
model.to("float16")
mod, named_params = model.export_tvm(spec=model.get_default_spec())
prefill_str = mod["prefill"].script()
print(*prefill_str.split("\n")[3:20], sep="\n")  # Only show the first 10 lines for demonstration
print("        ...")

print("\nParameters:")
pprint(named_params[:5])

@register_pipeline("opt_llm")
def _pipeline(
	ext_mods: List[nn.ExternModule] = None,
):
	ext_mods = ext_mods or []

	@tvm.transform.module_pass(opt_level=0)
	def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
		seq = tvm.transform.Sequential(
			[
				# Phase 1. Passes on high-level operator graph
				# We can enable cublas for further optimization
				relax.transform.FuseTransposeMatmul(),
				# Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
				relax.transform.LegalizeOps(),
				relax.transform.AnnotateTIROpPattern(),
				relax.transform.FoldConstant(),
				relax.transform.FuseOps(),
				relax.transform.FuseTIR(),
				# Phase 3. Passes on TIR
				relax.transform.DeadCodeElimination(),
				# Phase 4. Low-level Optimizations
				dlight.ApplyDefaultSchedule(
					dlight.gpu.Matmul(),
					dlight.gpu.GEMV(),
					dlight.gpu.Reduction(),
					dlight.gpu.GeneralReduction(),
					dlight.gpu.Fallback(),
				),
				# Phase 5. Lowering to VM bytecode
				relax.transform.RewriteDataflowReshape(),
				relax.transform.ToNonDataflow(),
				relax.transform.RemovePurityChecking(),
				relax.transform.CallTIRRewrite(),
				relax.transform.StaticPlanBlockMemory(),
				relax.transform.RewriteCUDAGraph(),
				relax.transform.LowerAllocTensor(),
				relax.transform.KillAfterLastUse(),
				relax.transform.LowerRuntimeBuiltin(),
				relax.transform.VMShapeLower(),
				relax.transform.AttachGlobalSymbol(),
				relax.transform.AttachExternModules(ext_mods),
			]
		)
		mod = seq(mod)
		return mod

	return _pipeline


with target:
	ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline("opt_llm"))
	vm = relax.VirtualMachine(ex, dev)

import numpy as np
import safetensors.torch
import torch

HF_WEIGHT_PATH = Path('TinyLlama-1.1B-Chat-v1.0') / "model.safetensors"

if not HF_WEIGHT_PATH.exists():
	url = 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors'
	urllib.request.urlretrieve(url, HF_WEIGHT_PATH)

param_dict = safetensors.torch.load_file(HF_WEIGHT_PATH , device="cpu")
# Numpy format weights
param_dict = {
	k: v.half().numpy() if v.dtype == torch.bfloat16 else v.numpy()
	for k, v in param_dict.items()
}

named_params = dict(named_params)
for i in range(model_config.num_hidden_layers):
	# Add QKV in self attention
	attn = f"model.layers.{i}.self_attn"
	param_dict[f"{attn}.qkv_proj.weight"] = np.concatenate(
		[
			param_dict.pop(f"{attn}.q_proj.weight"),  # Pop the old parameters to save memory
			param_dict.pop(f"{attn}.k_proj.weight"),
			param_dict.pop(f"{attn}.v_proj.weight"),
		],
		axis=0,
	)
	# Add gates in MLP
	mlp = f"model.layers.{i}.mlp"
	param_dict[f"{mlp}.gate_up_proj.weight"] = np.concatenate(
		[
			param_dict.pop(f"{mlp}.gate_proj.weight"),
			param_dict.pop(f"{mlp}.up_proj.weight"),
		],
		axis=0,
	)

# Convert params into ndarray
params = [
	tvm.nd.array(param_dict[k].astype("float16"), device=dev) for k in named_params.keys()
]