# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_topk_scores,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
        enable_reduce_sample: bool = False,
    ):
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.use_fp64_gumbel = use_fp64_gumbel
        self.enable_reduce_sample = enable_reduce_sample
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.req_states = req_states
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()

        if enable_reduce_sample and self.tp_size > 1:
            logger.info_once(
                "Reduced sampling is enabled for tensor-parallel inference "
                "(tp_size=%d). Each TP rank will perform local top-k and "
                "all-gather only the candidate values + indices instead of "
                "the full vocabulary logits.",
                self.tp_size,
            )

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(
            idx_mapping_np
        )
        return_logprobs = max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0

        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
        )

        if return_logprobs:
            if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
            logprobs_tensors = compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        if not self._requires_logits_processing(idx_mapping_np):
            return logits

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return True
        if np.any(self.penalties_state.use_penalty[idx_mapping_np]):
            return True
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True

        states = self.sampling_states
        temperatures = states.temperature.np[idx_mapping_np]
        if np.any((temperatures != 0.0) & (temperatures != 1.0)):
            return True
        if np.any(states.min_p.np[idx_mapping_np] != 0.0):
            return True
        if np.any(states.top_k.np[idx_mapping_np] != states.vocab_size):
            return True
        return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))

    # ------------------------------------------------------------------
    # Reduced sampling (tensor-parallel optimized)
    # ------------------------------------------------------------------

    def _reduce_sample(
        self,
        local_logits: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Vocab-parallel sampling via local top-k + TP all-gather.

        Each TP rank selects its top-k logits locally, all-gathers only those
        k values + global indices, then reuses ``apply_top_k_top_p`` and
        ``gumbel_sample`` on the reduced candidate set (treated as a "small
        vocabulary"). Communication: O(B * 2 * k * tp_size) instead of
        O(B * V).

        The same ``gumbel_sample`` path serves all-greedy, mixed, and
        all-random batches: greedy rows (temperature=0) skip Gumbel noise
        inside the kernel, falling back to plain argmax. ``all_greedy`` only
        affects ``k_for_topk`` (1 vs. user top_k) to minimize communication.

        Args:
            local_logits: Shard-local logits [num_tokens, V_local] (after
                temperature, penalties, etc. have been applied).
            top_k: Per-token top_k values [num_tokens] or None.
            top_p: Per-token top_p values [num_tokens] or None.
            expanded_idx_mapping: [num_tokens] mapping to request state idx.
            idx_mapping_np: [num_tokens] numpy mapping (for GPU-sync-free
                access to sampling states).
            temperature: [max_num_reqs] per-request temperature.
            seeds: [max_num_reqs] per-request random seeds.
            pos: [num_tokens] position within request (for RNG seeding).

        Returns:
            Sampled global token IDs [num_tokens].
        """
        _, V_local = local_logits.shape

        # All ranks must use the same k so all-gather shapes match.
        # Use numpy to avoid GPU-CPU synchronization.
        temps = self.sampling_states.temperature.np[idx_mapping_np]
        all_greedy = not np.any(temps != 0.0)
        if all_greedy:
            k_for_topk = 1
        elif top_k is not None:
            k_for_topk = min(
                int(self.sampling_states.top_k.np[idx_mapping_np].max()), V_local
            )
        else:
            k_for_topk = V_local
        k_for_topk = max(k_for_topk, 1)

        # Local top-k + global index mapping.
        local_vals, local_idx = torch.topk(local_logits, k=k_for_topk, dim=-1)
        local_global_idx = local_idx + self.tp_rank * V_local

        # All-gather only candidates.
        gathered_vals = tensor_model_parallel_all_gather(local_vals, dim=-1)
        gathered_idx = tensor_model_parallel_all_gather(local_global_idx, dim=-1)

        # Reuse apply_top_k_top_p + gumbel_sample on the candidate set.
        # gumbel_sample already handles greedy (temp=0) by skipping Gumbel
        # noise and falling back to argmax, so the same path serves both
        # all-greedy and mixed/random batches.
        # Clamp top_k to cand_size: rows with top_k = vocab_size (meaning
        # "no top_k") would otherwise produce negative gather indices when
        # cand_size < vocab_size.
        cand_size = gathered_vals.shape[-1]
        if top_k is not None:
            top_k = top_k.to(torch.long).clamp(min=1, max=cand_size).to(torch.int32)
        gathered_vals = apply_top_k_top_p(gathered_vals, top_k, top_p)
        sample_pos = gumbel_sample(
            gathered_vals,
            expanded_idx_mapping,
            temperature,
            seeds,
            pos,
            apply_temperature=False,
            use_fp64=self.use_fp64_gumbel,
        ).unsqueeze(-1)

        # Map candidate position → global token ID.
        return gathered_idx.gather(
            dim=-1, index=sample_pos
        ).squeeze(-1).to(torch.int64)

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )

        # ------------------------------------------------------------------
        # Reduced sampling path (tensor-parallel optimized).
        # When enabled and tp_size > 1, logits are shard-local [B, V_local]
        # (the all-gather in LogitsProcessor was skipped). We perform local
        # top-k, all-gather only candidates, then sample on the reduced set.
        # ------------------------------------------------------------------
        if self.enable_reduce_sample and self.tp_size > 1:
            top_k, top_p = self.sampling_states.get_top_k_top_p(
                expanded_idx_mapping, idx_mapping_np
            )
            sampled = self._reduce_sample(
                processed_logits,
                top_k,
                top_p,
                expanded_idx_mapping,
                idx_mapping_np,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
            )
            return sampled, processed_logits

        # ------------------------------------------------------------------
        # Standard sampling path (full-vocab logits).
        # ------------------------------------------------------------------
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np
        )
        use_flashinfer = self.use_flashinfer and not (
            # Don't use FI sampler if no requests use top_k/top_p, if there are
            # any greedy requests or per-request seeds, or if post-processed
            # logprobs need to be returned for any requests.
            (top_k is None and top_p is None)
            or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
            or self.sampling_states.any_greedy(idx_mapping_np)
            or self.sampling_states.any_explicit_seed(idx_mapping_np)
        )

        # Sample the next token.
        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            )
        return sampled, processed_logits
