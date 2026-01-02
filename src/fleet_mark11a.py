
# === Mark 11a — Full Stochastic Fusion Fleet Prototype (fixed PRNG handling + static n_mc_samples) ===
# Save as: fleet_mark11a_fixed2.py
# Run: python fleet_mark11a_fixed2.py

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
import jax.random as random

CORE, BUFFER, ELEC, BLANKET, STRUCT, BREED = 0, 1, 2, 3, 4, 5
N_STORES = 6

OP_FUSION     = jnp.uint8(0)
OP_DISPATCH   = jnp.uint8(1)
OP_DEC        = jnp.uint8(2)
OP_ALPHA      = jnp.uint8(3)
OP_NEUTRONQ   = jnp.uint8(4)
OP_BLANKET    = jnp.uint8(5)
OP_RECUP      = jnp.uint8(6)
OP_LOSSES     = jnp.uint8(7)
OP_CONDUCTION = jnp.uint8(8)
OP_CLAMP_HIGH = jnp.uint8(9)
OP_PROJECT    = jnp.uint8(10)

EV = 1.602176634e-19
MEV = 1e6 * EV
E_NEUTRON_J = 14.1 * MEV

class Stores(NamedTuple):
    capacity_J: jnp.ndarray
    level_J: jnp.ndarray
    temp_K: jnp.ndarray
    heatcapacity_J_per_K: jnp.ndarray
    neutron_queue_J: jnp.ndarray
    breed_store_J: jnp.ndarray

class ReactorParams(NamedTuple):
    dec_eff_base: jnp.float64
    alpha_frac_physical: jnp.float64
    blanket_eff: jnp.float64
    breed_rate: jnp.float64
    recup_eff_base: jnp.float64
    recup_throughput_J_per_step: jnp.float64
    base_loss_W: jnp.ndarray
    conductance_W_per_K: jnp.ndarray
    temp_min_K: jnp.ndarray
    temp_max_K: jnp.ndarray
    neutron_thermal_frac: jnp.float64
    neutron_struct_frac: jnp.float64
    recup_temp_scale: jnp.float64
    fidelity: jnp.int32
    neutron_transport_delay: jnp.int32
    control_temp_set: jnp.float64
    control_gain: jnp.float64
    control_smoothness: jnp.float64
    n_mc_samples: jnp.int32
    rng_seed: jnp.int32

class OpMeta(NamedTuple):
    elec_prod_J: jnp.float64
    elec_cred_J: jnp.float64
    loss_J: jnp.ndarray
    breed_J: jnp.float64

class AuditRecord(NamedTuple):
    op_id: jnp.uint8
    delta_J: jnp.ndarray
    meta: OpMeta

def meta_zero(stores: Stores) -> OpMeta:
    return OpMeta(jnp.float64(0.0), jnp.float64(0.0),
                  jnp.zeros_like(stores.level_J, dtype=jnp.float64), jnp.float64(0.0))

def enforce_K_symmetric_zero_diag(K: jnp.ndarray) -> jnp.ndarray:
    K_sym = (K + K.T) * jnp.float64(0.5)
    return K_sym - jnp.diag(jnp.diag(K_sym))

def safe_frac_mul(frac, J):
    return jnp.float64(frac) * J

def neutron_queue_push_pop(queue: jnp.ndarray, add_J: jnp.float64):
    ready = queue[0]
    tail = queue[1:]
    new_queue = jnp.concatenate([tail, jnp.reshape(add_J, (1,))], axis=0)
    return ready, new_queue

def op_fusion_enqueue(stores: Stores, Ffusion_W: jnp.ndarray, dt: jnp.float64, params: ReactorParams):
    J_fusion = jnp.sum(Ffusion_W) * dt
    J_alpha = safe_frac_mul(params.alpha_frac_physical, J_fusion)
    J_neutron = J_fusion - J_alpha
    delta = jnp.zeros_like(stores.level_J).at[CORE].add(J_alpha)
    meta = meta_zero(stores)
    return AuditRecord(OP_FUSION, delta, meta), J_fusion, J_neutron, J_alpha

def dispatch_controller_scale(T_core: jnp.float64, params: ReactorParams):
    x = params.control_smoothness * (params.control_temp_set - T_core)
    scale = 1.0 / (1.0 + jnp.exp(x))
    scale = jnp.clip((1.0 - params.control_gain) + params.control_gain * scale, 0.0, 1.0)
    return scale

def op_dispatch_prioritized(stores: Stores, flows_dst: jnp.ndarray, eff_mod: jnp.ndarray,
                            Fdispatch_W: jnp.ndarray, dt: jnp.float64, control_scale: jnp.float64):
    J_cmd = Fdispatch_W * dt * control_scale
    is_buffer = (flows_dst == BUFFER)
    is_elec = (flows_dst == ELEC)

    J_cmd_elec = jnp.where(is_elec, J_cmd, 0.0)
    want_elec = jnp.sum(J_cmd_elec)
    avail = stores.level_J[CORE]

    elec_scale = jnp.minimum(1.0, avail / (want_elec + 1e-18))
    allocated_elec = J_cmd_elec * elec_scale
    used_by_elec = jnp.sum(allocated_elec)
    avail_after_elec = avail - used_by_elec

    J_cmd_buffer = jnp.where(is_buffer, J_cmd, 0.0)
    total_buffer_req = jnp.sum(J_cmd_buffer)
    buffer_scale = jnp.minimum(1.0, avail_after_elec / (total_buffer_req + 1e-18))
    allocated_buffer_raw = J_cmd_buffer * buffer_scale

    eff = jnp.clip(eff_mod, 0.0, 1.0)
    buf_add_per_flow = jnp.where(is_buffer, eff * allocated_buffer_raw, 0.0)
    allocated_buffer_used = jnp.sum(allocated_buffer_raw)
    used_total = used_by_elec + allocated_buffer_used

    delta = jnp.zeros_like(stores.level_J)
    delta = delta.at[CORE].add(-used_total)
    delta = delta.at[BUFFER].add(jnp.sum(buf_add_per_flow))

    elec_th_per_flow = allocated_elec
    J_th_elec_total = jnp.sum(elec_th_per_flow)

    flows_losses_vec = jnp.zeros_like(stores.level_J)
    flows_losses_vec = flows_losses_vec.at[CORE].set(jnp.sum(jnp.where(is_buffer, (1.0 - eff) * allocated_buffer_raw, 0.0)))

    meta = OpMeta(jnp.float64(0.0), jnp.float64(0.0), flows_losses_vec, jnp.float64(0.0))
    return AuditRecord(OP_DISPATCH, delta, meta), J_th_elec_total, elec_scale, buffer_scale

def op_dec_adaptive(stores: Stores, J_th_elec_total: jnp.float64, params: ReactorParams, credit=True):
    T_core = stores.temp_K[CORE]
    gain = 1.0 / (1.0 + jnp.exp(-params.recup_temp_scale * (T_core - jnp.float64(700.0))))
    dec_eff = jnp.clip(params.dec_eff_base * (jnp.float64(0.6) + jnp.float64(0.4) * gain), 0.0, 1.0)
    Je = dec_eff * J_th_elec_total
    flag = jnp.where(credit, 1.0, 0.0)
    delta = jnp.zeros_like(stores.level_J).at[ELEC].add(Je * flag)
    loss_vec = jnp.zeros_like(stores.level_J).at[CORE].set(J_th_elec_total * (1.0 - dec_eff))
    meta = OpMeta(Je, Je * flag, loss_vec, jnp.float64(0.0))
    return AuditRecord(OP_DEC, delta, meta), Je, dec_eff

def op_blanket_stochastic_capture(stores: Stores, J_neutrons_ready: jnp.float64, params: ReactorParams, key, n_mc_samples: int):
    # n_mc_samples is expected to be a Python int and static for JIT
    n = n_mc_samples
    J_av = J_neutrons_ready

    def sample_capture(key):
        unifs = random.uniform(key, shape=(n,), dtype=jnp.float64)
        samples = (unifs < params.blanket_eff).astype(jnp.float64)
        capture_frac = jnp.mean(samples)
        return capture_frac

    key1, _ = random.split(key)
    capture_frac = jnp.where(J_av <= 0.0, 0.0, sample_capture(key1))
    J_capture = capture_frac * J_av
    J_leak = jnp.maximum(0.0, J_av - J_capture)

    immediate = params.neutron_thermal_frac * J_capture
    struct_part = params.neutron_struct_frac * J_capture
    residual = J_capture - immediate - struct_part
    immediate = immediate + residual

    delta = jnp.zeros_like(stores.level_J)
    delta = delta.at[BLANKET].add(immediate)
    delta = delta.at[STRUCT].add(struct_part)

    J_breed = params.breed_rate * J_capture
    loss_vec = jnp.zeros_like(stores.level_J).at[BLANKET].set(J_leak)
    meta = OpMeta(jnp.float64(0.0), jnp.float64(0.0), loss_vec, jnp.float64(J_breed))
    return AuditRecord(OP_BLANKET, delta, meta), J_capture, J_leak, J_breed

def op_recup_thermal(stores: Stores, params: ReactorParams, dt: jnp.float64):
    J_blanket_av = stores.level_J[BLANKET]
    T_blank = stores.temp_K[BLANKET]
    temp_factor = jnp.clip((T_blank - 300.0) / 900.0, 0.0, 1.0)
    recup_eff = params.recup_eff_base * (jnp.float64(0.5) + jnp.float64(0.5) * temp_factor)
    J_can_extract = recup_eff * J_blanket_av
    J_extract = jnp.minimum(J_can_extract, params.recup_throughput_J_per_step * dt)
    J_struct_av = stores.level_J[STRUCT]
    J_struct_recov = jnp.minimum(params.recup_eff_base * 0.1 * J_struct_av, params.recup_throughput_J_per_step * dt - J_extract)
    J_struct_recov = jnp.maximum(jnp.float64(0.0), J_struct_recov)
    total_recup = J_extract + J_struct_recov

    delta = jnp.zeros_like(stores.level_J)
    delta = delta.at[BLANKET].add(-J_extract)
    delta = delta.at[STRUCT].add(-J_struct_recov)
    delta = delta.at[BUFFER].add(total_recup)
    meta = meta_zero(stores)
    return AuditRecord(OP_RECUP, delta, meta), total_recup, recup_eff

def op_base_losses(stores: Stores, base_loss_W: jnp.ndarray, dt: jnp.float64):
    J_req = base_loss_W * dt
    J_av = stores.level_J
    J_loss = jnp.minimum(J_req, J_av)
    deficit = J_req - J_loss
    delta = -J_loss
    meta = OpMeta(jnp.float64(0.0), jnp.float64(0.0), J_loss, jnp.float64(0.0))
    return AuditRecord(OP_LOSSES, delta, meta), deficit

def op_event(stores: Stores, event_loss_W: jnp.float64, event_elec_W: jnp.float64, dt: jnp.float64, credit=True):
    J_req_loss = event_loss_W * dt
    J_req_e = event_elec_W * dt
    J_av = stores.level_J[CORE]
    total_req = J_req_loss + J_req_e
    scale = jnp.minimum(1.0, J_av / (total_req + 1e-18))
    J_loss = J_req_loss * scale
    J_e_evt = J_req_e * scale
    flag = jnp.where(credit, 1.0, 0.0)
    delta = jnp.zeros_like(stores.level_J)
    delta = delta.at[CORE].add(-(J_loss + J_e_evt))
    delta = delta.at[ELEC].add(J_e_evt * flag)
    deficit = total_req * (1.0 - scale)
    loss_vec = jnp.zeros_like(stores.level_J).at[CORE].set(J_loss + J_e_evt * (1.0 - flag))
    meta = OpMeta(J_e_evt, J_e_evt * flag, loss_vec, jnp.float64(0.0))
    return AuditRecord(OP_LOSSES, delta, meta), deficit

def op_conduction_energy(stores: Stores, K: jnp.ndarray, dt: jnp.float64):
    K_eff = enforce_K_symmetric_zero_diag(K)
    T = stores.temp_K
    Q = K_eff * (T[:, None] - T[None, :]) * dt
    delta = -jnp.sum(Q, axis=1)
    return AuditRecord(OP_CONDUCTION, delta, meta_zero(stores))

def op_clamp_high(stores: Stores, raw_level: jnp.ndarray, E_max: jnp.ndarray):
    clamp_J = jnp.maximum(0.0, raw_level - E_max)
    delta = -clamp_J
    meta = OpMeta(jnp.float64(0.0), jnp.float64(0.0), clamp_J, jnp.float64(0.0))
    return AuditRecord(OP_CLAMP_HIGH, delta, meta), clamp_J

def op_conservation_project(stores: Stores, new_level: jnp.ndarray, total_losses: jnp.ndarray,
                            J_fusion: jnp.float64, neutron_queue_sum_old: jnp.float64, neutron_queue_sum_new: jnp.float64, breed_sum_old: jnp.float64, breed_sum_new: jnp.float64):
    old_sum = jnp.sum(stores.level_J) + neutron_queue_sum_old + breed_sum_old
    new_sum = jnp.sum(new_level) + neutron_queue_sum_new + breed_sum_new
    cons = new_sum - old_sum + jnp.sum(total_losses) - J_fusion
    tol = 1e-6

    eligible = new_level > (1e-12)
    eligible_f = eligible.astype(jnp.float64)
    any_eligible = jnp.any(eligible)
    eligible_f = jnp.where(any_eligible, eligible_f, jnp.ones_like(eligible_f))

    weights = eligible_f * jnp.maximum(new_level, 0.0)
    total_weights = jnp.sum(weights) + 1e-18
    delta_proj = -cons * (weights / total_weights)
    apply = jnp.where(jnp.abs(cons) > tol, 1.0, 0.0)
    delta_proj = delta_proj * apply

    proj_meta = OpMeta(jnp.float64(0.0), jnp.float64(0.0), jnp.zeros_like(stores.level_J), jnp.float64(0.0))
    proj_rec = AuditRecord(OP_PROJECT, delta_proj, proj_meta)
    corrected_level = new_level + delta_proj
    return proj_rec, corrected_level, cons

def summarize_ops(ops: list):
    def summarize(op: AuditRecord):
        total = jnp.sum(op.delta_J)
        elec_p = op.meta.elec_prod_J
        elec_c = op.meta.elec_cred_J
        losses = jnp.sum(op.meta.loss_J)
        breed = op.meta.breed_J
        return jnp.stack([total, elec_p, elec_c, losses, breed])
    return jnp.stack([summarize(op) for op in ops])

def one_step(stores: Stores, params: ReactorParams, Ffusion_W: jnp.ndarray, Fdispatch_W: jnp.ndarray,
             flows_dst: jnp.ndarray, eff_mod: jnp.ndarray, dt: jnp.float64, key, n_mc_samples: int, credit=True,
             event_loss_W=jnp.float64(0.0), event_elec_W=jnp.float64(0.0), debug_assert=False):
    ops = []

    fusion_rec, J_fusion, J_neutron_new, J_alpha = op_fusion_enqueue(stores, Ffusion_W, dt, params)
    ops.append(fusion_rec)

    ready_neutrons, new_queue = neutron_queue_push_pop(stores.neutron_queue_J, J_neutron_new)
    nq_delta = jnp.zeros_like(stores.level_J)
    nq_meta = meta_zero(stores)
    nq_rec = AuditRecord(OP_NEUTRONQ, nq_delta, nq_meta)
    ops.append(nq_rec)

    T_core = stores.temp_K[CORE]
    control_scale = dispatch_controller_scale(T_core, params)

    dispatch_rec, J_th_elec_total, elec_scale, buffer_scale = op_dispatch_prioritized(stores, flows_dst, eff_mod, Fdispatch_W, dt, control_scale)
    ops.append(dispatch_rec)

    dec_rec, Je, dec_eff = op_dec_adaptive(stores, J_th_elec_total, params, credit)
    ops.append(dec_rec)

    key_b, key_rest = random.split(key)
    blanket_rec, J_capture, J_leak, J_breed = op_blanket_stochastic_capture(stores, ready_neutrons, params, key_b, n_mc_samples)
    ops.append(blanket_rec)

    recup_rec, J_recup, recup_eff = op_recup_thermal(stores, params, dt)
    ops.append(recup_rec)

    alpha_rec = AuditRecord(OP_ALPHA, jnp.zeros_like(stores.level_J), meta_zero(stores))
    ops.append(alpha_rec)

    base_rec, base_deficit = op_base_losses(stores, params.base_loss_W, dt); ops.append(base_rec)
    cond_rec = op_conduction_energy(stores, params.conductance_W_per_K, dt); ops.append(cond_rec)
    event_rec, event_deficit = op_event(stores, jnp.float64(event_loss_W), jnp.float64(event_elec_W), dt, credit); ops.append(event_rec)

    total_delta = jnp.zeros_like(stores.level_J)
    total_losses = jnp.zeros_like(stores.level_J)
    total_breed_credit = jnp.float64(0.0)
    for op in ops:
        total_delta = total_delta + op.delta_J
        total_losses = total_losses + op.meta.loss_J
        total_breed_credit = total_breed_credit + op.meta.breed_J

    raw_level = stores.level_J + total_delta
    raw_breed = stores.breed_store_J + total_breed_credit

    E_min = stores.heatcapacity_J_per_K * params.temp_min_K
    E_max = stores.heatcapacity_J_per_K * params.temp_max_K
    clamp_rec, clamp_J = op_clamp_high(stores, raw_level, E_max); ops.append(clamp_rec)
    total_losses = total_losses + clamp_rec.meta.loss_J

    new_level = raw_level - clamp_J

    neutron_queue_sum_old = jnp.sum(stores.neutron_queue_J)
    neutron_queue_sum_new = jnp.sum(new_queue)
    breed_sum_old = jnp.sum(stores.breed_store_J)
    breed_sum_new = jnp.sum(raw_breed)
    proj_rec, corrected_level, cons = op_conservation_project(stores, new_level, total_losses, J_fusion, neutron_queue_sum_old, neutron_queue_sum_new, breed_sum_old, breed_sum_new)
    ops.append(proj_rec)

    new_temp = corrected_level / jnp.maximum(stores.heatcapacity_J_per_K, 1e-12)
    new_breed_store = raw_breed

    new_stores = Stores(
        stores.capacity_J,
        corrected_level,
        new_temp,
        stores.heatcapacity_J_per_K,
        new_queue,
        new_breed_store
    )

    viol_floor = corrected_level < E_min

    audit = {
        "Jfusion": J_fusion,
        "Jneutron_enqueued": J_neutron_new,
        "Jneutron_ready": ready_neutrons,
        "Jalpha": J_alpha,
        "Jcapture": J_capture,
        "Jleak": J_leak,
        "Jbreed_credit": total_breed_credit,
        "Jrecup": J_recup,
        "Jthelec": J_th_elec_total,
        "Jeprod": Je,
        "dispatch_elec_scale": elec_scale,
        "dispatch_buffer_scale": buffer_scale,
        "dec_eff": dec_eff,
        "recup_eff": recup_eff,
        "Jclamphigh": jnp.sum(clamp_J),
        "violfloor": jnp.any(viol_floor),
        "basedeficit": jnp.sum(base_deficit),
        "eventdeficit": event_deficit,
        "conservation_residual": cons
    }

    return new_stores, ops, audit

@partial(jax.jit, static_argnames=['horizon','n_modules','n_mc_samples'])
def fleet_rollout(initial_modules: Stores, params: ReactorParams,
                  flows_dst: jnp.ndarray, eff_mod: jnp.ndarray,
                  Ffusion_series_W: jnp.ndarray, Fdispatch_series_W: jnp.ndarray,
                  dt: jnp.float64, horizon: int, n_modules: int, n_mc_samples: int, credit=True,
                  event_loss_W=jnp.float64(0.0), event_elec_W=jnp.float64(0.0), base_key=None):
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_modules), initial_modules)

    def step(carry, i):
        st_mods = carry
        Ffusion_t = Ffusion_series_W[i]
        Fdispatch_t = Fdispatch_series_W[i]

        key_step = random.fold_in(base_key, i)
        keys_mod = random.split(key_step, n_modules)

        def per_mod(st, Ffus_mod, Fdis_mod, key_mod):
            new_st, ops, audit = one_step(st, params, Ffus_mod, Fdis_mod, flows_dst, eff_mod, dt, key_mod, n_mc_samples, credit, event_loss_W, event_elec_W, debug_assert=False)
            ops_summary = summarize_ops(ops)
            return new_st, (audit, ops_summary)

        new_states, (audits, ops_summaries) = jax.vmap(per_mod)(st_mods, Ffusion_t, Fdispatch_t, keys_mod)

        fleet_Jfusion = jnp.sum(audits["Jfusion"])
        fleet_Jneutron_enq = jnp.sum(audits["Jneutron_enqueued"])
        fleet_Jneutron_ready = jnp.sum(audits["Jneutron_ready"])
        fleet_Jthelec = jnp.sum(audits["Jthelec"])
        fleet_Je = jnp.sum(audits["Jeprod"])
        fleet_scale_avg = jnp.mean(audits["dispatch_elec_scale"])
        fleet_Jclamp_high = jnp.sum(audits["Jclamphigh"])
        fleet_viol_floor = jnp.any(audits["violfloor"])
        fleet_deficit = jnp.sum(audits["basedeficit"]) + jnp.sum(audits["eventdeficit"])
        fleet_breed = jnp.sum(audits["Jbreed_credit"])
        fleet_cons_resid = jnp.sum(audits["conservation_residual"])

        max_temp = jnp.max(new_states.temp_K)
        sum_levels = jnp.sum(new_states.level_J)
        sum_neutron_queue = jnp.sum(new_states.neutron_queue_J)
        sum_breed_store = jnp.sum(new_states.breed_store_J)

        telemetry = jnp.stack([
            fleet_Jfusion, fleet_Jneutron_enq, fleet_Jneutron_ready, fleet_Jthelec, fleet_Je, fleet_scale_avg,
            fleet_Jclamp_high, jnp.float64(fleet_viol_floor), fleet_deficit, fleet_breed, fleet_cons_resid,
            max_temp, sum_levels, sum_neutron_queue, sum_breed_store
        ])
        return new_states, (new_states, telemetry, ops_summaries)

    final_states, outs = jax.lax.scan(step, states, jnp.arange(horizon, dtype=jnp.int32))
    states_seq, telemetry_seq, ops_summaries_seq = outs
    return final_states, states_seq, telemetry_seq, ops_summaries_seq

if __name__ == "__main__":
    import numpy as np
    import csv

    D = 6

    stores = Stores(
        jnp.array([6e10, 5e10, 1.2e10, 5e9, 1e9, 1e9], dtype=jnp.float64),
        jnp.array([3.6e10, 2.4e10, 6e9, 2.5e9, 5e8, 1e8], dtype=jnp.float64),
        jnp.array([600.0, 500.0, 400.0, 350.0, 320.0, 300.0], dtype=jnp.float64),
        jnp.array([6e7, 3.5e7, 2.5e7, 5e6, 2e6, 1e6], dtype=jnp.float64),
        jnp.zeros((D,), dtype=jnp.float64),
        jnp.array([1e8], dtype=jnp.float64)
    )

    params = ReactorParams(
        jnp.float64(0.82),
        jnp.float64(0.20),
        jnp.float64(0.94),
        jnp.float64(0.05),
        jnp.float64(0.06),
        jnp.float64(5e9),
        jnp.array([2e6, 8e5, 4e5, 5e5, 2e5, 1e5], dtype=jnp.float64),
        jnp.array([
            [0.0, 9e4, 6e4, 2e4, 1e4, 5e3],
            [9e4, 0.0, 5e4, 1e4, 5e3, 2e3],
            [6e4, 5e4, 0.0, 3e3, 2e3, 1e3],
            [2e4, 1e4, 3e3, 0.0, 1e3, 5e2],
            [1e4, 5e3, 2e3, 1e3, 0.0, 2e2],
            [5e3, 2e3, 1e3, 5e2, 2e2, 0.0]
        ], dtype=jnp.float64),
        jnp.array([300.0, 300.0, 300.0, 300.0, 300.0, 300.0], dtype=jnp.float64),
        jnp.array([1200.0, 1000.0, 800.0, 700.0, 600.0, 500.0], dtype=jnp.float64),
        jnp.float64(0.7),
        jnp.float64(0.25),
        jnp.float64(0.006),
        jnp.int32(2),
        jnp.int32(D),
        jnp.float64(600.0),
        jnp.float64(0.5),
        jnp.float64(0.02),
        jnp.int32(256),
        jnp.int32(42)
    )

    flows_dst = jnp.array([BUFFER, ELEC, ELEC, CORE], dtype=jnp.int32)
    eff_mod = jnp.array([0.92, 0.90, 0.88, 0.85], dtype=jnp.float64)

    n_modules = 8
    horizon = 200
    dt = jnp.float64(0.1)
    credit = True

    t = np.arange(horizon, dtype=np.float64)
    base_dispatch = np.array([9.0e9, 7.5e9, 3.0e9, 2.0e9], dtype=np.float64)
    mod_scale = (1.0 + 0.12 * np.sin(2.0 * np.pi * t / horizon)).reshape(horizon, 1, 1)
    fleet_base = np.tile(base_dispatch, (n_modules, 1)).astype(np.float64)
    Fdispatch_series = (mod_scale * fleet_base).astype(np.float64)

    fusion_pulse = np.zeros((horizon, n_modules, 1), dtype=np.float64)
    fusion_pulse[12:25, :, 0] = 6e9
    fusion_pulse[110:121, :, 0] = 1.5e10

    Fdispatch_series = jnp.array(Fdispatch_series)
    Ffusion_series = jnp.array(fusion_pulse)

    base_key = random.PRNGKey(int(params.rng_seed))

    # Pass n_mc_samples as concrete Python int
    n_mc_samples = int(params.n_mc_samples)

    final_states, states_seq, telemetry_seq, ops_summaries_seq = fleet_rollout(
        stores, params, flows_dst, eff_mod,
        Ffusion_series, Fdispatch_series,
        dt, horizon, n_modules, n_mc_samples, credit=credit,
        event_loss_W=jnp.float64(0.0), event_elec_W=jnp.float64(0.0),
        base_key=base_key
    )

    csv_path = "fleetmark11a_audit_fixed2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step","fleetJfusion","fleetJneutron_enq","fleetJneutron_ready","fleetJthelec","fleetJeprod","fleetdispatchscale_avg",
            "fleetJclamphigh","fleetviolfloor","fleetdeficit","fleetbreed","fleetconsresid","fleetmaxtempK","fleetsumlevelsJ","fleetneutronqueueJ","fleetbreedstoreJ"
        ])
        tele = np.array(telemetry_seq)
        for i in range(horizon):
            writer.writerow([
                int(i),
                float(tele[i,0]), float(tele[i,1]), float(tele[i,2]), float(tele[i,3]), float(tele[i,4]),
                float(tele[i,5]), float(tele[i,6]), int(tele[i,7]), float(tele[i,8]), float(tele[i,9]),
                float(tele[i,10]), float(tele[i,11]), float(tele[i,12]), float(tele[i,13]), float(tele[i,14])
            ])
    print("CSV audit ditulis ke:", csv_path)

    # Host audit (including BREED physical store)
    ops_last = np.array(ops_summaries_seq[-1])
    fleet_ops_last = np.sum(ops_last, axis=0)
    explicit_losses_last = float(jnp.sum(fleet_ops_last[:,3]))

    init_total_E_per_module = float(np.sum(np.array(stores.level_J))) + float(np.sum(np.array(stores.neutron_queue_J))) + float(np.sum(np.array(stores.breed_store_J)))
    init_total_E = init_total_E_per_module * float(n_modules)

    final_total_E = float(np.sum(np.array(final_states.level_J))) + float(np.sum(np.array(final_states.neutron_queue_J))) + float(np.sum(np.array(final_states.breed_store_J)))

    delta_total_E = final_total_E - init_total_E

    def losses_series_accumulate(ops_summaries_seq_np):
        series = []
        for step in ops_summaries_seq_np:
            fleet_ops = np.sum(step, axis=0)
            losses = float(np.sum(fleet_ops[:, 3]))
            series.append(losses)
        return np.array(series, dtype=np.float64)

    ops_seq_np = np.array(ops_summaries_seq)
    losses_series = losses_series_accumulate(ops_seq_np)
    total_losses_horizon = float(np.sum(losses_series))

    tele_np = np.array(telemetry_seq)
    fleet_Jfusion_series = tele_np[:,0]
    total_fusion = float(np.sum(fleet_Jfusion_series))

    final_neutron_queue_sum = float(jnp.sum(np.array(final_states.neutron_queue_J)))
    final_breed_store_sum = float(jnp.sum(np.array(final_states.breed_store_J)))

    temp_from_energy = (np.array(final_states.level_J) / np.array(stores.heatcapacity_J_per_K))
    temp_consistency_max = float(np.max(np.abs(temp_from_energy - np.array(final_states.temp_K))))

    conservation = delta_total_E + total_losses_horizon - total_fusion

    print("=== Mark 11a Audit Ledger (full stochastic, breed physical, fixed PRNG + static MC) ===")
    print("Explicit losses total (last step):", explicit_losses_last)
    print("Total energy change ΔEtotal (fleet):", delta_total_E)
    print("Σ(losses over horizon):", total_losses_horizon)
    print("Σ(fusion over horizon):", total_fusion)
    print("Conservation check (ΔE_total + Σ(losses) - Σ(fusion)):", conservation)
    print("Final neutron queue total (fleet):", final_neutron_queue_sum)
    print("Final breed store total (fleet):", final_breed_store_sum)
    print("Thermal consistency max |T(U)-T|:", temp_consistency_max)
