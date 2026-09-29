# P1 structural-neighborhood sensitivity

This directory freezes the outcome-blind Stage-A predicate and labels, followed by the separate fixed-prediction Stage-B audit. Stage A reads only the public host/guest maps and the non-outcome fields used by the predicate. Stage B reads the frozen labels and already released predictions; it does not fit a model.

The historical filename `predicate_v2_1.py` is retained because it is part of the frozen file binding; its header identifies the executed v2.2 rule. `HOST_L3` is the extended sulfonated-calixarene host definition: 32 labelled hosts, including the 17-member `HOST_L2` subset and 15 additional labelled hosts. The additional set contains name-labelled calix[5], calix[6], calix[8], thia-calix[4], and bis-calix[4] entries. These are frozen labels, not a new name-based reclassification.

The supplied Stage-B report is the saved historical result. To replay it into a fresh directory, set `SUPRABENCH_P1_OUTPUT` if desired and run `python protocols/p1_d6sc_overlap/stage_b_reaggregate.py`. Stage-A execution remains separate so that outcome-side results cannot feed back into the predicate.
