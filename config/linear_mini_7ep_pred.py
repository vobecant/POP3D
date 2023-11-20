_base_ = [
    '/home/vobecant/PhD/TPVFormer-OpenSet/out/tpv04_small_occupancy_mini_wFeats_agnostic.py/tpv04_small_occupancy_mini_wFeats_agnostic.py',
]

model_params = dict(
    input_dim=384,
    hidden_dim=384,
    num_hidden=2,
    nbr_class=18
)