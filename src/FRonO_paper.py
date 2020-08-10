from conf import FRonO_KMEANS_EXP_DIR


def scratch_steps(elec_collection, evt_collection):
    from soz_predictor import evt_rate_soz_pred_baseline_localized
    evt_rate_soz_pred_baseline_localized(elec_collection,
                                         evt_collection,
                                         intraop=False,
                                         restrict_to_tagged_coords=True,
                                         restrict_to_tagged_locs=True,
                                         evt_types_to_load=['Fast RonO'],
                                         evt_types_to_cmp=[['Fast RonO']],
                                         locations={0: ['Whole Brain']},
                                         saving_dir=FRonO_KMEANS_EXP_DIR,
                                         # return_pat_dic_by_loc=True,
                                         plot_rocs=False,
                                         remove_elec_artifacts=True)
