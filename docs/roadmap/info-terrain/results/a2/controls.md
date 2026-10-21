# Control measurements

Revision A-2 draft 3; native habitat analysis; [2,20) s; soundgen 3.0.0 aggregation. Hypotheses recorded without acceptance assertions.

| Stimulus | Original S / precision | Extended S / precision |
|---|---:|---:|
| 1_sine | 6.666231491081105e-12 / 0.9999999999997858 | 2.0895979797924038e-11 / 0.9853299124549723 |
| 2a_white | 0.543955423429962 / 0.8474364779845751 | 0.47002444200108245 / 0.8952138269059791 |
| 2b_white_quiet | 0.5439554238982626 / 0.8474364777915679 | 0.4700244434291709 / 0.8952138271324083 |
| 3_pink | 0.57463121736582 / 0.8463548280435795 | 0.3989665919450907 / 0.9346969988734904 |
| 4_periodic | 0.5389393762273472 / 0.511118892450274 | 0.36547959333693764 / 0.9633399845294492 |
| 5_deviant | 0.722013635511968 / 0.4932798835705016 | 0.3441338805353113 / 0.8949326138779751 |

Gain pair implementation check in the registered [2,20) s window: passed. Full masks and indicator values compared at every bin and hop; startup diagnostics retained separately in JSON.

Additional startup diagnostic: 4 tolerance violations over the full [0,20) s input; see exact times and bins in controls.json. Unregistered exact binary gain (1/32) probe: 0 tolerance violations.
