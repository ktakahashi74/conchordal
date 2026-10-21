# External comparison

Revision A-2 draft 3; 303 prepared WAVs; 300 stimuli in the common set; 25 prototypes. Bootstrap: 10000 shared cluster resamples, splitmix64 seed 20261020. S reused from A-1; O uses input-unit perceptual floors.

soundgen_primary

| Version | r | 95% CI |
|---|---:|---|
| R-S | 0.5324889295652043 | [0.3771790958937368,0.664132088102627] |
| R-O | 0.5094495548593614 | [0.38171999699325515,0.6284201549990812] |
| N-S | 0.49303345480827493 | [0.3350108619541946,0.6284761765233771] |
| N-O | 0.5418606587486092 | [0.38940697602828445,0.6586639877608389] |
| N-P-S | 0.27374058242543026 | [0.10926231541579567,0.3979822428420154] |
| N-P-O | 0.3035265497163697 | [0.11594357512018778,0.45208311574440435] |

- R-S minus N-S: 0.0394554747569294, 95% CI [-0.0935291831138409,0.1706664527977375]
- N-S minus N-P-S: 0.21929287238284467, 95% CI [0.1317997845834274,0.3129990074539852]
- R-S minus R-O: 0.02303937470584294, 95% CI [-0.025941677210115418,0.07376441226729473]

osf_auxiliary

| Version | r | 95% CI |
|---|---:|---|
| R-S | 0.598205831969605 | [0.45608336636517194,0.7195355799516661] |
| R-O | 0.5902845999514098 | [0.4716424049105053,0.6951321117033765] |
| N-S | 0.6259383404049624 | [0.4894374831954265,0.7312386399996506] |
| N-O | 0.6413077925072941 | [0.5351986404934135,0.7234973338758522] |
| N-P-S | 0.4347301746176767 | [0.28968283547507073,0.5604095735427724] |
| N-P-O | 0.5055487976458085 | [0.3573287619616495,0.6335730791690466] |

- R-S minus N-S: -0.02773250843535746, 95% CI [-0.13709286611606158,0.08732210889113022]
- N-S minus N-P-S: 0.19120816578728572, 95% CI [0.12970311638637888,0.24618399816959405]
- R-S minus R-O: 0.007921232018195168, 95% CI [-0.029325795289698947,0.047303309980579904]

Exclusions (per version):

- R-S: [{"file":"vig_01.wav","reasons":["token_join_failure"],"token":"01"},{"file":"vig_25.wav","reasons":["token_join_failure"],"token":"25"},{"file":"vig_75.wav","reasons":["token_join_failure"],"token":"75"}]
- R-O: [{"file":"vig_01.wav","reasons":["token_join_failure"],"token":"01"},{"file":"vig_25.wav","reasons":["token_join_failure"],"token":"25"},{"file":"vig_75.wav","reasons":["token_join_failure"],"token":"75"}]
- N-S: [{"file":"vig_01.wav","reasons":["token_join_failure"],"token":"01"},{"file":"vig_25.wav","reasons":["token_join_failure"],"token":"25"},{"file":"vig_75.wav","reasons":["token_join_failure"],"token":"75"}]
- N-O: [{"file":"vig_01.wav","reasons":["token_join_failure"],"token":"01"},{"file":"vig_25.wav","reasons":["token_join_failure"],"token":"25"},{"file":"vig_75.wav","reasons":["token_join_failure"],"token":"75"}]
- N-P-S: [{"file":"vig_01.wav","reasons":["token_join_failure"],"token":"01"},{"file":"vig_25.wav","reasons":["token_join_failure"],"token":"25"},{"file":"vig_75.wav","reasons":["token_join_failure"],"token":"75"}]
- N-P-O: [{"file":"vig_01.wav","reasons":["token_join_failure"],"token":"01"},{"file":"vig_25.wav","reasons":["token_join_failure"],"token":"25"},{"file":"vig_75.wav","reasons":["token_join_failure"],"token":"75"}]

R is the registered Rust STFT/HTK-mel approximation, with symmetric Hann and raw one-sided band power. Leading MP3 silence retained. Fit uncertainty excluded from bootstrap. OSF auxiliary uses the registered per-hop maxima and population SD, rather than reproducing the bundled final-window weight reuse.
