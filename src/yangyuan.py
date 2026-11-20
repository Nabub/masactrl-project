"""Confidence-gated attention masks and hybrid initialization for MasaCtrl.

The benchmark mask stays with the evaluator. Editing only uses the source image,
prompts, and optional subject text. The source prompt is used for localization
while inversion and source denoising keep the baseline's empty conditioning.
"""

import re
import warnings

import torch
import torch.nn.functional as F


MASK_RESOLUTION = 16
MASK_THRESHOLD = 0.25
FOREGROUND_STRENGTH = 0.8
CONFIDENCE_MOMENTUM = 0.7
GENERIC_WORDS = frozenset(
    "a an the and or is are this that of to in on at for with by from as "
    "photo image picture portrait painting drawing view side front background".split()
)


def _clean_prompt(prompt):
    return prompt.replace("[", "").replace("]", "").strip()


def _content_words(prompt):
    return [word for word in re.findall(r"[\w'-]+", prompt.lower()) if word not in GENERIC_WORDS]


def _subject_tokens(tokenizer, source_prompt, target_prompt, subject_hint):
    """Resolve the annotated subject against each prompt's actual CLIP tokens."""
    source, target = _clean_prompt(source_prompt), _clean_prompt(target_prompt)
    end = min(tokenizer.model_max_length, 77) - 1

    def tokens(prompt):
        return tokenizer.encode(prompt)[1:-1][:end - 1]

    def match(prompt_ids, phrase):
        phrase_ids = tokenizer.encode(phrase, add_special_tokens=False) if phrase else []
        positions = set()
        if phrase_ids:
            for offset in range(len(prompt_ids) - len(phrase_ids) + 1):
                if prompt_ids[offset:offset + len(phrase_ids)] == phrase_ids:
                    positions.update(range(offset + 1, offset + 1 + len(phrase_ids)))
        return sorted(positions)

    target_ids = tokens(target)
    if not target_ids:
        raise ValueError("Yangyuan needs a nonempty target prompt")
    hint_words = _clean_prompt(subject_hint or "").split()
    annotated_target = re.findall(r"\[([^\]]+)\]", target_prompt)
    if not source:
        # The demo has an empty source prompt and no benchmark subject hint.
        target_candidates = [" ".join(hint_words[i:]) for i in range(len(hint_words))]
        target_candidates += list(reversed(_content_words(target)))
        for candidate in target_candidates:
            target_indices = match(target_ids, candidate)
            if target_indices:
                source = f"a photo of {candidate}"
                return source, target, match(tokens(source), candidate), target_indices
        return target, target, list(range(1, len(target_ids) + 1)), list(range(1, len(target_ids) + 1))

    source_ids = tokens(source)
    # PIE-Bench's blended_word joins the source and target subject phrases.
    # Try possible boundaries, preferring the midpoint for multi-word subjects.
    splits = sorted(range(1, len(hint_words)), key=lambda position: abs(2 * position - len(hint_words)))
    phrases = [(" ".join(hint_words[:i]), " ".join(hint_words[i:])) for i in splits]
    hint = " ".join(hint_words)
    phrases.append((hint, hint))
    target_words = set(_content_words(target))
    phrases.extend((word, word) for word in _content_words(source) if word in target_words)
    for source_phrase, target_phrase in phrases:
        source_indices = match(source_ids, source_phrase)
        target_indices = match(target_ids, target_phrase)
        if source_indices and target_indices:
            return source, target, source_indices, target_indices

    def first_match(prompt_ids, candidates):
        for candidate in candidates:
            indices = match(prompt_ids, candidate)
            if indices:
                return indices
        return list(range(1, len(prompt_ids) + 1))

    annotated_source = re.findall(r"\[([^\]]+)\]", source_prompt)
    return (
        source,
        target,
        first_match(source_ids, annotated_source + _content_words(source)),
        first_match(target_ids, annotated_target + _content_words(target)),
    )


def _normalize(values):
    least = values.amin(dim=(-2, -1), keepdim=True)
    span = values.amax(dim=(-2, -1), keepdim=True) - least
    return (values - least) / span.clamp_min(1e-8)


def _resize(values, side):
    return F.interpolate(values, (side, side), mode="bilinear", align_corners=False)


def _soft_mask(scores):
    width = 0.08
    weight = ((scores - MASK_THRESHOLD + width) / (2 * width)).clamp(0, 1)
    return weight.square() * (3 - 2 * weight)


class _AttentionState:
    def __init__(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = 0
        self.current_maps = {}

    def add_map(self, side, values):
        values = values.reshape(1, 1, side, side).detach()
        total, count = self.current_maps.get(side, (torch.zeros_like(values), 0))
        self.current_maps[side] = total + values, count + 1

    def averaged_maps(self):
        return {side: _normalize(total / count) for side, (total, count) in self.current_maps.items()}


class _InversionProbe(_AttentionState):
    """Collect 16x16 subject attention at the first eight inversion steps."""

    def __init__(self, embeddings, indices):
        super().__init__()
        self.embeddings = embeddings
        self.indices = indices
        self.projected_keys = {}
        self.priors = []

    def cross_attention(self, layer, q, probabilities, heads, side):
        if self.cur_step >= 8:
            return
        if q.shape[0] != 2 * heads:
            raise ValueError("Source probe requires one source with classifier-free guidance")
        if layer not in self.projected_keys:
            key = layer.to_k(self.embeddings)
            self.projected_keys[layer] = key.reshape(-1, heads, q.shape[-1]).transpose(0, 1)
        source_queries = q[heads:]
        logits = (source_queries @ self.projected_keys[layer].transpose(1, 2)) * layer.scale
        self.add_map(side, logits.softmax(-1)[..., self.indices].sum(-1).mean(0))

    def after_step(self):
        if self.current_maps:
            maps = self.averaged_maps()
            self.priors.append(maps[MASK_RESOLUTION].float().cpu())
            self.current_maps.clear()

    def source_mask(self, size):
        if not self.priors:
            raise RuntimeError("No source attention was recorded during inversion")
        prior = _normalize(torch.stack(self.priors).mean(0))
        return _soft_mask(_resize(prior, size))


class _ForegroundTransfer(_AttentionState):
    """Transfer source keys and values only where the target localizes the subject."""

    def __init__(self, source_mask, indices, start_step, start_layer, *, confidence_gate=True,
                 confidence_momentum=CONFIDENCE_MOMENTUM):
        super().__init__()
        self.source_mask = source_mask
        self.indices = indices
        self.start_step = start_step
        self.start_layer = start_layer
        self.confidence_gate = confidence_gate
        self.confidence_momentum = confidence_momentum
        self.previous_maps = {}
        self.source_masks = {}
        # Keep one confidence history per attention layer. The same layer is
        # visited at every denoising step, so this smooths time without mixing
        # confidence values from different U-Net resolutions.
        self.previous_confidence = {}

    def cross_attention(self, layer, q, probabilities, heads, side):
        if q.shape[0] != 4 * heads:
            raise ValueError("Foreground transfer expects a source/target CFG batch")
        target = probabilities[3 * heads:, :, self.indices].sum(-1).mean(0)
        self.add_map(side, target)

    def available_maps(self):
        maps = dict(self.previous_maps)
        for side, current in self.averaged_maps().items():
            maps[side] = 0.7 * current + 0.3 * maps[side] if side in maps else current
        return maps

    def after_step(self):
        self.previous_maps = self.available_maps()
        self.current_maps.clear()

    def self_attention(self, q, k, v, output, heads, scale, chunk_size, side):
        if self.cur_step < self.start_step or self.cur_att_layer // 2 < self.start_layer:
            return output
        maps = self.available_maps()
        if not maps or side * side != q.shape[1]:
            return output
        if side not in self.source_masks:
            self.source_masks[side] = F.interpolate(self.source_mask.to(q), (side, side), mode="area").flatten()
        source = self.source_masks[side]
        target = _soft_mask(_resize(maps[MASK_RESOLUTION], side)).reshape(1, -1, 1) * FOREGROUND_STRENGTH
        if not source.any().item() or not target.any().item():
            return output
        bias = source.clamp_min(1e-6).log().masked_fill(source <= 0, torch.finfo(q.dtype).min)
        # CFG layout: unconditional source/target, then conditional source/target.
        target_q = q.reshape(4, heads, *q.shape[1:])[[1, 3]].flatten(0, 1)
        source_k = k.reshape(4, heads, *k.shape[1:])[[0, 2]].flatten(0, 1)
        source_v = v.reshape(4, heads, *v.shape[1:])[[0, 2]].flatten(0, 1)
        if self.confidence_gate:
            transferred, confidence = _attention_with_confidence(
                target_q, source_k, source_v, scale, chunk_size, bias,
                valid_keys=(source > 0).sum().clamp_min(2),
            )
            layer_index = self.cur_att_layer
            previous = self.previous_confidence.get(layer_index)
            if previous is not None and previous.shape == confidence.shape:
                momentum = self.confidence_momentum
                confidence = momentum * previous + (1 - momentum) * confidence
            confidence = confidence.clamp(0, 1)
            self.previous_confidence[layer_index] = confidence.detach()
        else:
            transferred = _attention(target_q, source_k, source_v, scale, chunk_size, bias)
            confidence = torch.ones(
                (target_q.shape[0], target_q.shape[1], 1), device=target_q.device, dtype=target_q.dtype
            )
        transfer_gate = target * confidence
        routed = output.reshape(4, heads, *output.shape[1:]).clone()
        transfer_gate = transfer_gate.reshape(2, heads, output.shape[1], 1)
        routed[[1, 3]] = routed[[1, 3]] * (1 - transfer_gate) + transferred.reshape(
            2, heads, *output.shape[1:]
        ) * transfer_gate
        return routed.flatten(0, 1)


def _attention(q, k, v, scale, chunk_size, key_bias=None):
    """Bound memory by splitting queries, giving every query all keys."""
    output = torch.empty_like(q)
    transposed_k = k.transpose(1, 2)
    for start in range(0, q.shape[1], chunk_size):
        stop = start + chunk_size
        scores = torch.bmm(q[:, start:stop], transposed_k) * scale
        if key_bias is not None:
            scores = scores + key_bias
        output[:, start:stop] = torch.bmm(scores.softmax(-1), v)
    return output


def _attention_with_confidence(q, k, v, scale, chunk_size, key_bias=None, valid_keys=None):
    """Chunked attention plus per-query confidence from normalized entropy.

    A concentrated distribution over source foreground keys is treated as a
    reliable correspondence. Diffuse attention falls back toward target-native
    features. Confidence is returned per batch/head/query as [B, Q, 1].
    """
    output = torch.empty_like(q)
    confidence = torch.empty((*q.shape[:2], 1), device=q.device, dtype=torch.float32)
    transposed_k = k.transpose(1, 2)
    if valid_keys is None:
        valid_keys = k.shape[-2]
    log_valid_keys = torch.log(torch.as_tensor(valid_keys, device=q.device, dtype=torch.float32)).clamp_min(1e-6)
    for start in range(0, q.shape[1], chunk_size):
        stop = min(start + chunk_size, q.shape[1])
        scores = torch.bmm(q[:, start:stop], transposed_k) * scale
        if key_bias is not None:
            scores = scores + key_bias
        probabilities = scores.softmax(-1)
        output[:, start:stop] = torch.bmm(probabilities, v)
        probabilities_fp32 = probabilities.float()
        entropy = -(probabilities_fp32 * probabilities_fp32.clamp_min(1e-8).log()).sum(-1, keepdim=True)
        confidence[:, start:stop] = (1 - entropy / log_valid_keys).clamp(0, 1)
    return output, confidence


def _register_attention(model, editor, chunk_size, *, baseline=False):
    if chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    cross_probe = getattr(editor, "cross_attention", None)
    foreground_transfer = getattr(editor, "self_attention", None)

    def make_forward(layer):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            if attention_mask is not None or mask is not None:
                raise NotImplementedError("External attention masks are unsupported in this SD 1.4 runner")
            context = encoder_hidden_states if encoder_hidden_states is not None else context
            is_cross = context is not None
            context = context if is_cross else x
            batch, tokens, _ = x.shape
            heads = layer.heads

            def split(projected):
                return projected.reshape(batch, -1, heads, projected.shape[-1] // heads).transpose(1, 2)

            q, k, v = split(layer.to_q(x)), split(layer.to_k(context)), split(layer.to_v(context))
            if (baseline and not is_cross and editor.cur_step in getattr(editor, "step_idx", ())
                    and editor.cur_att_layer // 2 in editor.layer_idx):
                if batch != 4:
                    raise ValueError("MasaCtrl requires a source/target CFG batch")
                k, v = k[[0, 0, 2, 2]], v[[0, 0, 2, 2]]
            q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
            side = int(tokens ** 0.5)
            if is_cross:
                probabilities = (torch.bmm(q, k.transpose(1, 2)) * layer.scale).softmax(-1)
                if cross_probe is not None and tokens == MASK_RESOLUTION ** 2:
                    cross_probe(layer, q, probabilities, heads, side)
                output = torch.bmm(probabilities, v)
            else:
                output = _attention(q, k, v, layer.scale, chunk_size)
                if foreground_transfer is not None:
                    output = foreground_transfer(q, k, v, output, heads, layer.scale, chunk_size, side)
            editor.cur_att_layer += 1
            if editor.cur_att_layer == editor.num_att_layers:
                editor.cur_att_layer = 0
                editor.cur_step += 1
                editor.after_step()
            output = output.reshape(batch, heads, tokens, -1).transpose(1, 2).reshape(batch, tokens, -1)
            projection = layer.to_out[0] if isinstance(layer.to_out, torch.nn.ModuleList) else layer.to_out
            return projection(output)
        return forward

    layers = [layer for block in (model.unet.down_blocks, model.unet.mid_block, model.unet.up_blocks)
              for layer in block.modules() if layer.__class__.__name__ == "Attention"]
    if not layers:
        raise RuntimeError("No Diffusers 0.15 attention modules found to register")
    editor.num_att_layers = len(layers)
    for layer in layers:
        layer.forward = make_forward(layer)


def register_chunked_baseline(model, editor, query_chunk_size=128):
    """Enable the same memory-bounded attention math for the baseline runner."""
    _register_attention(model, editor, query_chunk_size, baseline=True)


@torch.no_grad()
def improved_edit(model, source_tensor, source_prompt, target_prompt, *, subject_hint="",
                  steps, guidance, start_step, start_layer, seed, query_chunk_size=128,
                  confidence_gate=True, confidence_momentum=CONFIDENCE_MOMENTUM):
    """Run the improved MasaCtrl method and return the edited 3×H×W tensor."""
    if guidance <= 1:
        raise ValueError("Yangyuan needs classifier-free guidance greater than 1")
    probe_prompt, target_prompt, source_indices, target_indices = _subject_tokens(
        model.tokenizer, source_prompt, target_prompt, subject_hint
    )
    probe_input = model.tokenizer(probe_prompt, padding="max_length", max_length=77,
                                  truncation=True, return_tensors="pt")
    embeddings = model.text_encoder(probe_input.input_ids.to(source_tensor.device))[0]
    probe = _InversionProbe(embeddings, source_indices)
    _register_attention(model, probe, query_chunk_size)
    inversion, _ = model.invert(source_tensor, "", guidance_scale=guidance,
                                num_inference_steps=steps, return_intermediates=False)

    size = source_tensor.shape[-1]
    source_mask = probe.source_mask(size)
    gate = F.interpolate(source_mask.to(inversion), inversion.shape[-2:], mode="area")
    if (gate > 0).any() and (gate == 0).any():
        noise_device = "cpu" if inversion.device.type == "mps" else inversion.device
        rng = torch.Generator(device=noise_device).manual_seed(seed)
        noise = torch.randn(inversion.shape, generator=rng, device=noise_device,
                            dtype=inversion.dtype).to(inversion.device)
        target_latent = inversion * gate + noise * (1 - gate)
    else:
        warnings.warn("Attention mask did not separate foreground and background. Using source initialization.",
                      RuntimeWarning, stacklevel=2)
        target_latent = inversion

    if not 0 <= confidence_momentum < 1:
        raise ValueError("confidence_momentum must be in [0, 1)")
    editor = _ForegroundTransfer(
        source_mask, target_indices, start_step, start_layer,
        confidence_gate=confidence_gate, confidence_momentum=confidence_momentum,
    )
    _register_attention(model, editor, query_chunk_size)
    result = model(["", target_prompt], latents=torch.cat([inversion, target_latent]),
                   num_inference_steps=steps, guidance_scale=guidance, height=size, width=size)
    return result[-1]
