| Field | Type | Description |
| --- | --- | --- |
| body | enum/union |  |
| perceptual | enum/union |  |
| phonation | enum/union |  |
| pitch | enum/union |  |


## BodyMethod

Enum values: sine, harmonic

## BodyPatch

| Field | Type | Description |
| --- | --- | --- |
| amp | null or number |  |
| method | enum/union |  |
| timbre | enum/union |  |


## PerceptualPatch

| Field | Type | Description |
| --- | --- | --- |
| adaptation | null or number |  |
| enabled | bool or null |  |
| novelty_bias | null or number |  |
| self_focus | null or number |  |


## PhonationPatch

| Field | Type | Description |
| --- | --- | --- |
| density | null or number |  |
| legato | null or number |  |
| sociality | null or number |  |
| sync | null or number |  |
| type | enum/union |  |


## PhonationType

Enum values: clock, field, hold, interval, none


## PitchMode

Enum values: free, lock

## PitchPatch

| Field | Type | Description |
| --- | --- | --- |
| mode | enum/union |  |
| freq | null or number | Frequency in Hz: center for free mode, fixed output for lock. |
| range_oct | null or number |  |
| gravity | null or number |  |
| exploration | null or number |  |
| persistence | null or number |  |


## TimbrePatch

| Field | Type | Description |
| --- | --- | --- |
| brightness | null or number |  |
| inharmonic | null or number |  |
| motion | null or number |  |
| width | null or number |  |

