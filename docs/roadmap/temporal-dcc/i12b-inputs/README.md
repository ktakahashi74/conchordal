# I12b入力・取得順の草案

このディレクトリは[I12b実装前登録](../i12b-coupling.md)の入力・実行順を具体化する。登録状態の正本は別途作る`freeze.json`、実行可能なcandidate source／binary／判定器の正本は取得前の別seal。これらがそろうまで実行しない。現時点で取得なし。

## 入力と版

`inputs/`のpulse付きRhai 6件は[I11-2凍結入力](../i11-2-inputs/plan.json)のbyteコピー。48 kHz、seed 20260924、6秒＋終端0.5秒、habitat専用2 Hz pulseとpresentation専用の選択対象Voiceを変えない。旧12条件の`None`回帰は[I11-1登録済みRhai](../i11-inputs/inputs/)を`plan.json`のpath・SHAで参照し、再コピーしない。十ファイルのSHAと旧12のSHAを`plan.json`へ保存した。

四つのconfigの意味は次のとおり。`full`と`none`はI11-2の`config-body-on.toml`、`config-none.toml`とbyte一致。`forward-cut`は`full`の`arrival_weight = 1.0`を`0.0`へ一箇所だけ変更し、`arrival = true`を維持する。`backward-cut`は`full`の`[temporal_period]`に`period_accent_input = false`を一行だけ追加する案。これ以外のTOML値と旧Hazard係数・horizon・body設定は不変。旧12条件のseedは20260918、新6条件は20260924。Hazard係数・means・deviations・horizonの全`f64` bitを`plan.json`へ固定した。最後のキーは実装前なので、現binaryでのparse成功を主張しない。

TOML本文のprofile file欄には旧登録のLinux上のpathが残る。Mac取得時は旧I11-2と同様、SHA `033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`の実ファイルを確認した後、実行用configコピーのfile欄だけを置換する。原本・登録SHAを変更せず、元／派生configのSHA、profile元／実path・SHA、file以外のparse値一致を取得provenanceへ保存する。旧binaryと新binaryには同じ派生`full`を渡す。

旧binaryはI11-2の[封印済みcandidate manifest](../../../../target/i11-stage2-20260924/candidate/manifest.json)で同定する。manifest SHAは`e3b76d987ddb4b7a479d5ac3927d2c27d2064af1a40f229a4f51fa971e4d45f4`。新binaryは取得前に別manifestへsource fileごとのSHA、`Cargo.toml`／lock、build command・toolchain、render／instrument binary SHA、検査器SHAを封印する。`plan.json`の新manifest値は意図して`null`のままにし、後日のsealで値を確定する。封印後にplanを書き換えない。旧I11-2のsource・入力・ledgerへ書き込まない。

新field契約は、`period_accent_input`の既定`true`をserialize時に省略し、旧`full`/`None`のreport形を保つ。B腕の`false`だけ`temporal_observation.observation.period_parameters.period_accent_input`に出す。新規に受領した実accentのperiod寄与を切った数は`temporal_observation.observation.period_accent_cut_deliveries`とprofile `background.shared[0].period_accent_cut_deliveries`へB腕だけ累積表示する。full/forward-cut/Noneでは両欄を省略する。旧F対新Fと`None`のbit回帰に**新metadataの除外欄は0件**。旧I11-2が既に除外した実時間欄などのみ継承し、予期しない新fieldは凍結前に原因を確認する。

## 実行順と判定

`run_order`はsequence 1–48のofflineと49–192の性能。offlineの最初12本は旧12条件の新binary `None`／report回帰。続く36本は旧pulse6条件×3腕×report有無。case順はI11-2 `new_pulse_cases`の登録順、mode順はreport→no-report、腕順はcase/mode indexによる固定回転。report有無のWAVも腕内で比較する。offlineだけ`conchordal-render`を使う。

性能144本は同じpulse6×report有無×3反復×4変種。変種は封印旧binaryの`old-full`、新binaryの`full`／`forward-cut`／`backward-cut`。反復→case→modeの順に取り、各cell内の四変種は`(rep−1 + case_index + mode_index) mod 4`で回転する。instrumentの`conchordal`は音声をdiskへ書かない。全runは逐次で、bin／config／scenario／profile・command・exit・report/profile/WAV/logのbyte数とSHAを台帳へ残す。AC給電・蓋開放・sleep抑止を開始時と各run前後に検査し、OS sleep logも保存する。統制違反や失敗のrawは保持し、結果を選別して再取得しない。

前向き作用は`full`と`forward-cut`が同じ到来publication/CDFを受け取り、weightだけの差から決定・WAV差に至るか検査する。後向き作用は、同じ実accent列からperiod pair/peakが変わり、`periodic=false`による群除外またはCDF値の変化、候補費用、決定、WAVへ至る**通常実走の総効果**を検査する。さらに同一case・同時刻の実`full`／`backward-cut` packetを、full側で凍結した非到来`TemporalParticipation`入力へ差し替える**制御直接効果**を別gateで検査する。packetのidentity、group系譜、受領時刻、期限、profile key、自己除外bindingを書き換えず、対応が無い組を報告する。実走の切断前prefix違反を直接効果で救済しない。B側の既知Pは必須にしないが、未知化だけを成功としない。三身体それぞれ同一caseで、総効果の決定・WAV差と直接効果の費用・選択差を要する案。

性能のcut間は各10欄の符号付き差を全240欄で報告し、同等合否へ使わない。非退行は新`full`対旧`old-full`の120欄**すべての合格**を要する。I11-2の固定A/A floor SHA `ac737802286730f020cb62dbd01220397f484883adcf66c149edbc599990c00f`を工学幅として使い、時間の中央値／p99は`new−base <= max(0.05×base, floor)`、最大は`new−base <= max(1,000 µs, floor)`、超過hop数は`new−base <= floor`。各run全hopの中央値／p99は昇順の`floor(q×(N−1))`番目、時間欄は各run統計の3反復中央値、超過件数は3反復合計。旧floor付き両側規則とfloorなし元規則も全欄併記する。reportなしの適格性は、旧`old-full`・新`full`・`forward-cut`ではprofile `background.arrival_known_decisions > 0`、Bでは`background.shared[0].period_accent_cut_deliveries > 0`を各case/mode/repで要求する。B側に既知Pを強制しない。旧I11-2原試行11欄不合格は変えず、R2・64 Voice・実機の全受入を主張しない。取得前にこの式、判定器、入力、run順を封印する。
