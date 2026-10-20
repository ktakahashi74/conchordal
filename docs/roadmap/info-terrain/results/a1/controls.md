# Control measurements

Draft 5; native habitat analysis; [2,20) s; soundgen 3.0.0 aggregation. Hypotheses recorded without acceptance assertions.

| Stimulus | Original S / precision | Extended S / precision |
|---|---:|---:|
| 1_sine | 0.5113506918944689 / 0.9999999999997858 | 0.03896419713484093 / 0.9997678476842996 |
| 2a_white | 0.543955423429962 / 0.8474364779845751 | 0.5033108299981098 / 0.894259269896757 |
| 2b_white_quiet | 0.5439554238982626 / 0.8474364777915679 | 0.5033108302447512 / 0.894259269519579 |
| 3_pink | 0.57463121736582 / 0.8463548280435795 | 0.5016896613239288 / 0.9371947802080178 |
| 4_periodic | 0.5389393762273472 / 0.511118892450274 | 0.38903665887301186 / 0.603531948208096 |
| 5_deviant | 0.722013635511968 / 0.4932798835705016 | 0.36780926506192335 / 0.5110959152282785 |

Gain pair implementation check in the registered [2,20) s window: passed. Full masks and indicator values compared at every bin and hop; startup diagnostics retained separately in JSON.

Additional startup diagnostic: 4 tolerance violations over the full [0,20) s input; see exact times and bins in controls.json. Unregistered exact binary gain (1/32) probe: 0 tolerance violations.
