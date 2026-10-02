# I11 第2段・試験用到来入力の登録

日付: 2026-09-26。状態: 取得前の試験用登録。`i11-onset-comparison.md` §4.5 の第2段を数値検査可能にする。通常既定、作者採用、I11-2 技術完了は決めない。

## 固定入力と数値参照

Hazard 試験入力は [`i11-stage2-hazard-test.toml`](i11-stage2-hazard-test.toml) の全バイトとし、SHA-256 は `a7993ce065a52e156a19ef82c97559842184674c3562bdd4b65d6c7e06e12485`。`c[0]=-2`、`c[10]=4`、他の16係数0、8個の平均0・偏差1、`horizon_sec=4.0`。この係数は `i13-author-controls.md` の「主登録の `c0=-2,c10=4` は未適合診断値」を試験用へ固定したもの。観測データへの適合、音楽上の推奨、既定採用を意味しない。`[temporal_onset_comparison]` は別設定であり、既定の `arrival=false` と `arrival_weight=1.0` を変更しない。

発行時計は48 kHzを試験範囲とし、発行時 sample を点0、960 sample刻みで点0〜200の201点を配る。点200は192,000 sample＝4.0秒。Hazard の経過時間が単一値で `reset_unknown=false`、かつ周期 peak がある場合だけ表を作る。発行時の peak／word／履歴を固定し、各20 ms区間の `arrival::Engine` の hazard 積分を累積する。`H_0=0`、`H_k=H_{k-1}+∫[e+(k-1)0.02,e+k0.02] hazard(u)du`、`F_k=-expm1(-H_k)`。1区間でも積分が未解決なら群の表全体を不明にする。`0≤F_k≤1` と単調性を検査し、異常値を部分的に採用しない。4秒の一括積分は、位相付き最短周期0.125秒に対して現行16分割の最大幅0.25秒となり、既存の周期の4分の1以下という収束条件を満たせない。

独立参照は [`scripts/i11_stage2_cdf_reference.py`](../../../scripts/i11_stage2_cdf_reference.py) の固定中点則（各20 msを256分割）であり、SHA-256 は `9047df485963324d8a1802cb22c5dbe4ec67517335b74754e9cc2228f7b7b9b1`。生産側の2点 Gauss–Legendre と別の計算経路を使う。定数2/秒の閉形式、位相依存と単調性、0／4秒端点、半格子補間、地平外、経過時間の二値不一致とreset不明、既知0と不明、Periodic の階段を [`tests/test_i11_stage2_cdf_reference.py`](../../../tests/test_i11_stage2_cdf_reference.py) で検査する。この参照は実装前に作った検査素材。Rust 側は試験実装で全201格子点を絶対差 `1e-5`、窓差分を `2e-5` で照合した。最大計算量は200区間×62 hazard評価×7群／busを上限として R2 予備計測へ渡す。

## 共有 snapshot と消費境界

`observation::Snapshot` にある `period.groups` の群 handle／forecast と、同じ snapshot の `group_prototypes.assignments[prototype]` を参照する。後者の `Assignment.key=(group.epoch,group.generation)` と snapshot bus で handle を復元する。`action_profile_features.groups[prototype]` は profile・gesture の双方があるときだけ作られるため、I11-2 自己群除外の必須入力にしない。存在する場合は同じ handle と一致検査する。Voice の `consumer::Binding` の source id・source generation・body generation、bus、model version、end／available、prototypeを照合して、同じ群を候補集合から除く。bindingまたは対応が無いときは除外しないで理由を記録する。この対応は descriptor による関連付けであり、完全な自声検出ではない。

群集合と費用は `i11-onset-comparison.md` §4.5 の式を維持する。`issued_at≤now<horizon_end`、periodic、habitat bus、有効な CDF を満たす群を、全候補に共通の集合として固定する。候補の `at+width` が地平外に出る群は集合から除き、`F(at±width)` を線形補間する。集合が空なら到来項は不明で費用へ加えず、既知の `P=0` と分ける。集合が非空なら `coupling×arrival_weight×(1-P)` を候補費用へ加え、skip 判定からは除く。現行 `Forecast::valid_for` は `cut≤horizon_end` を許すが、この消費側は原登録の厳密な `now<horizon_end` を使う。

実装前に残る技術選択は、共有 assignment と Binding の鮮度判定に I10 と同じ24,000 sample（48 kHzで0.5秒）を使うか、I11-2 専用の期限を定義するか、48 kHz以外を拒否するか sample 丸めを定めるか、Rust と独立参照の誤差許容値をいくつにするか、到来項を切る対照の作用差をどの母集団で数えるかである。これらは取得前に固定する。係数や `arrival_weight` を通常作者操作へ恒久採用するか、可聴差を受け入れるかは別の作者判断。試験用数値の固定、独立参照作成、境界検査にはその判断を要しない。

## 試験実装の追加固定（取得前、2026-09-26）

隔離実験の数値経路は48 kHzのみ。20 msを960 sample、4秒を192,000 sampleとして丸めなしで扱う。他の sample rate は unsupported とし、0確率を代入しない。独立参照との比較許容はCDF全201点それぞれ絶対差 `1e-5` 以下、線形補間後の到来窓確率は絶対差 `2e-5` 以下。期間内の20 ms積分が一つでも解決しなければ表全体を不明とする。過去の `Forecast.probability` が4秒一括積分で未解決でも、発行時の経過時間が単一値で各区間積分が解決すれば、この区間累積表は作る。値の都合で既知0に置き換えない。

自己群の照合は、同じ source id／source generation／body generation の Binding と、同じ bus・epoch・body model version の共有 assignment を使う。`Binding.end≤Binding.available≤now`、`assignment.end_sample≤period.received_at≤now` を要求し、`now−Binding.end<24,000`、`now−assignment.end_sample<24,000` sampleを鮮度期限とする。対象群の CDF 発行後に得た Binding／assignment では自己群を推定せず、`Binding.available≤CDF.issued_at` と `assignment.end_sample≤CDF.issued_at` を追加で要求する。`assignment.key` の epoch・generation が period 側の実在する群 handle と一致する場合だけ除外する。識別・支持時刻・モデル版が欠けた値を補わない。対応が不明なら自己群除外を行わず、`binding_absent`、`assignment_absent`、`identity_mismatch`、`assignment_future`、`assignment_stale`、`assignment_unmapped`、`period_future`、`group_absent`、`cdf_absent`、`assignment_after_cdf`、`profile_mismatch` の区別を decision record に残す。

一回の候補選択では、候補時刻の全列を先に確定し、`issued_at≤now<horizon_end`、habitat bus、periodic、CDF既知、自己群でないことを満たす群から、どれか一候補の `at+width` が地平を超える群を除く。この後の群集合を全候補で共用し、候補ごとに群数を変えない。群0件は到来項不明、非空集合で `P=0` は既知0。到来項だけを切る比較の分母・合否は、独立参照と配送を通した後、取得前に追記する。これは試験作用の登録であり、既定の `arrival=false` を変えない。

decision record は音群slotごとの集合除外理由を保持する。`cdf_unavailable` は現在の snapshot だけからは経過時間不確実・積分未解決・表未生成を区別できないため、その内部原因の断定には使わない。理由別内訳が §4.5 の完了条件に必要なら、生成側の欠測原因を別途配送する。unsupported rate／horizon、period欠測／未来時刻は群0件の具体的な全体状態として残す。

両モデルとも許容する。ただし `period_parameters.model` と群の `Forecast.model` の一致を必須とし、`Forecast::valid_for` には一致確認後のモデルを渡す。skip 判定では到来項の間接作用も除くため、全候補の到来項を除いた基底費用の最小値を閾値と比較する。選択後の候補から到来項だけを引いた費用では、到来により高基底費用の候補が選ばれると skip が変わる。この試験仕様は §4.5 の「到来項を閾値比較に含めない」を強く実現する解釈であり、取得前に固定した。両方の費用を decision record に残す。

## 到来ON/OFFの作用比較登録（取得前、2026-09-26）

素材は I11-1 の固定seed付き `sine-flow-4`、`harmonic-flow-4`、`modal-flow-4`。各 `.rhai` の SHA-256 は順に `655a2a8358a82678f7326ba2d128b2e3def3bafd4cb21b92db70cbf41a8e44f1`、`7c71543034976732d37ce5d7216f8eeac628a3c2d595fe3ad1be5b04b2d7b6a2`、`4608a6c9f1d5d02ef8d6ce056816a8a5ae412373ad63128868b6b8a5f73514f6`。試験用設定は [`i11-stage2-inputs/config-off.toml`](i11-stage2-inputs/config-off.toml) と [`config-on.toml`](i11-stage2-inputs/config-on.toml)。各SHA-256は `612ea263553c2f180e46fb28f9fb0c09cf30d5f4054fd78792087d7c829a60b3`、`791a210c366eb432727d6f124075f343b50dc5b5206837f724555aea5c744a3b`。I11-1 `config-proxy.toml` の全設定を保ち、`temporal_period` だけ上記試験Hazardへ置換し、双方とも `arrival_weight=4.0` を明示して `arrival` 真偽だけを変える。4.0は作用検出の試験用上限であり通常既定・作者採用ではない。実行前に各入力hashとsource・binaryのhashを保存する。

分母は素材ごとの全 `participation_decision` 件数と、ON/OFFの時刻・Voice IDが一致する**初回選択またはskip分岐までの共通接頭列**の件数を別々に示す。その接頭列で ON が `arrival_state=known`、群数正、全候補の到来確率既知となる決定を「到来作用を判定できる件数」とする。群0件、CDF欠測、時刻・候補配列の不一致は分母から黙って消さず、理由別件数として報告する。初回分岐が無い素材も全件比較して「作用差なし」と報告する。初回分岐後は音声と観測履歴が変わるため、後続差を到来項だけに帰属しない。

接頭列から初回分岐まで、時刻・Voice ID、候補23slotの有無とoffset、候補の時刻・変位・文脈・重なりと外部energy、非到来の入力（footprint識別・16 power・配送時刻・幅・forecast時刻・自己音・記憶）を厳密比較する。各候補の `ON.cost−ON.arrival_cost` と `OFF.cost` の差は丸めを考慮した絶対 `1e-5` 以下とする。初回分岐は ON が到来既知かつ選択 offset が異なる場合だけ到来項の作用証拠とし、その時点で skip だけが異なる場合、先行入力が異なる場合、未知の表を既知0とした場合は不合格。分岐素材の WAV hash 相違も確認する。3素材のうち少なくとも1素材でこの作用証拠が成立することを局所作用の合格基準とする。3素材の結果だけで母集団一般化や可聴受入は主張しない。

この判定器 [`scripts/compare_i11_stage2_arrival.py`](../../../scripts/compare_i11_stage2_arrival.py) の取得前SHA-256は `de7e8df4ec4e56b059001437ba5d4ad54b003067a855df1210677ec7f315e99b`。独立の小例testは [`tests/test_compare_i11_stage2_arrival.py`](../../../tests/test_compare_i11_stage2_arrival.py)（SHA-256 `cb9b06173f0e82b3cee235bc55c52fdb6bcaaa47c1079a340dfbec2c94ab2ec5`）で、非到来入力差と未知状態の偽作用を拒否する。

既定OFF回帰は同じ3素材の元の `config-none.toml`（SHA-256 `b9b93fb3e1cadfbd85bde87ea9223f23d4f795393b61ed874c5b1384e4f920f0`）を変更せず実行し、`06a4772c43d06b41b44753bb0be891f23e16b93e` 取得 `target/i11-same-record-20260926-193414/` の同名WAVと、I11-1 §5.4aの学習record 10種を bit 比較する。基準binary SHA-256は `eb852e34e78a9e0ec40465065ae12ca09fe818ecace08d8002375133ee3623fa`。候補recordは共通keyの決定的内容を別照合し、配送飽和による片側だけの件数を別記する。ON/OFFの作用比較とこの `None` 回帰は別判定。

## 試験実装の局所検証（登録比較の前）

隔離checkout `.worktrees/i11-stage2` に限る。Python独立参照5件、RustのCDF格子／窓／境界、共通群集合・自己群対応、消費選択とskip反例の単体testは合格。全 `cargo test` は `test_status.txt` に exit 0、通常の `cargo clippy -- -D warnings` も合格。`cargo clippy --all-targets -- -D warnings` は既存テストコードの警告で未合格であり、今回追加したコードの2警告は修正した。

`target/i11-stage2-smoke-20260926/` の固定seed・12秒の局所renderでは、試験Hazard設定と `arrival=true` でCDF付きperiod群609件、participation decision 51件中到来既知43件・群なし8件。`arrival=false` 対照との最初の選択差は `now=178176` sampleで、ONの `selected_at=189729.0121214282`、OFFの `selected_at=184929.0121214282`。OFFのreportに到来欄は無い。両WAVのSHA-256も異なる。これらは配線と局所作用の証拠であり、登録母集団の効果、A1作者採用、R2資源受入、既定採用の証拠ではない。

## 追加実装登録: CDF未生成の原因配送（2026-09-26）

最初の3素材比較のsourceと結果は保存したまま、隔離版に診断だけを追加する。
CDF数値、到来費用、候補、乱数、モデル係数、未知時の動作は変更しない。
生成関数を `Result` にし、未対応sample rate／horizon、発行・モデル・source identity不一致、
reset不明、経過時間不確実／不正、周期根拠なし、区間積分未解決、非有限・非単調なCDFを区別する。
発行forecast自体がない場合はその状態を記録し、CDF計算を実行して失敗したとは扱わない。

群snapshotにoptionalな未生成理由を運び、候補群の除外記録へ同じ理由を渡す。
到来項無効時には計算せず、追加snapshot欄もserializeしない。
既存の `cdf_unavailable` は理由を持たない入力のfallbackとして残すが、生成原因を推測しない。
成功時の201値・window値が独立参照に一致し続けること、原因別の拒否、群snapshotから実消費者への
理由配送、OFFのserialize不変を検査する。旧3素材の成功数値と比較して新しい診断以外の作用差を混ぜない。
I11 §5.7専有取得中は静的実装のみ行い、build/test/renderは取得終了後へ送る。
