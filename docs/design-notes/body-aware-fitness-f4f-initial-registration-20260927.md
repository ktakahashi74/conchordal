# F4f 初回 Field spawn の身体評価: 固定登録（2026-09-27）

## 対象と既存契約

第十二版の ecology 隔離 worktree で、初回 `Action::Spawn` の `SpawnStrategy::Field` にオフラインの試験専用評価入力を接続する。通常 source の既定分岐は変更しない。対象は `FieldSampling::Peak` と `Density` の Consonance/Dissonance/Edge。F4b/F4e と同じ72-hop代表身体、同じ Rust の `body_fitness::evaluate`、実際の子と同じ `VoiceSpec::spawn_with_landscape` 条件を使う。通常 runtime の出生評価配線、Hereditary、音色遺伝、実時間費用は対象外。

現行 `Community::decide_frequency` は指定周波数域を Log2Space の近傍bin範囲に切り、各bin中心を範囲内へ clamp する。Peakは非占有bin中の最大値を優先し、すべて占有なら全bin中の最大値を選ぶ。Densityは占有binの重みを0にして `WeightedIndex` で選ぶ。合計massが0なら非占有binの一様重み、それもなければ全binの一様重みへ戻す。選んだbinだけ最大16回の連続log周波数 jitterを試し、spacingを満たせなければclamp済みbin中心へ戻す。解析範囲との交差がない場合と Uniform target は指定域内のlog一様抽選を使う。この順序、clamp、占有、zero fallback、連続jitterは試験で保持する。

## 身体を使う数値と配置

候補binごとに、予約子ID、population ID、member index、世代0、現frame、Community seed、実template、候補周波数、出生前共有Landscapeから仮の子Voiceを作る。仮Voiceの身体と正規化代表Recipeから候補密度 `q_x(b)` を取得し、ERB幅による正規化質量 `w_x(b)` と、有効環境Cの身体平均 `fitness_score(x)` / `fitness_level(x)` を算出する。候補の同一性は周波数bit、子ID、世代、body、Recipe hash、frame、環境epoch・支持時刻で照合する。同じclamp周波数が複数bin slotに出た場合も、slotと選択重みを統合しない。RecipeはVoice側の位相0代表契約を使う。

Peakの Consonance/Dissonance/Edge はそれぞれ `fitness_level(x)`、`-fitness_level(x)`、`-|fitness_level(x)-0.5|` を比較する。Consonanceのtensionは全候補の `fitness_score` 最小・最大から目標scoreを作り、その目標との差を比較する。Gapは空き領域という別の目的なので、点の `subjective_intensity` による既存順位を保つ。Uniformも従来のlog一様配置を保つ。

Densityの Consonance では、`fitness_score` や `fitness_level` をmassへ転用しない。既存の `consonance_density_mass_eff(b)` は `rho` を含む `max(0, H01(b)*(1-rho*R01(b)))` のhabituation後の量なので、候補の非負massを `Σ_b w_x(b) consonance_density_mass_eff(b)` とする。これは一bin純音で既存massに一致し、`rho` の役割を維持する。Dissonance/Edgeも既存の各bin density mass関数を `w_x` で平均してから選択する案とする。Gapは現行のrange内最大強度を基準にした空隙massを保持する。Consonanceのtension係数は候補の身体 `fitness_score` とrange内目標scoreの距離から求め、上記massへ乗じる。各重みは非負・有限でなければunsupportedとし、0 massのfallbackは既存順序を維持する。

bin中心での身体評価と、選択後の連続jitterで確定する実子Recipeは別の候補になる。最終周波数の子を再構築してbody・Recipe identityを照合し、最終周波数におけるfitnessも別に算出する。binの選択重みが中心由来であること、最終子のfitnessが別の値になり得ることを結果へ明記する。最終子fitnessを選択に反映する方式は、全binのjitter抽選順と乱数列を変更するため、今回の登録範囲に入れない。

## オフライン注入と検査

取得条件を実行前に固定する。Community seed 7、48 kHz、512 samples/hop、初回機会 frame 64、population ID 7、子ID 2と3、世代0。range は380–520 Hz、解析 Log2Space は80–4000 Hz・48 bins/octave、代表観測72 hop、解析epoch 37。template は440 Hzの固定ratio Harmonic、Entrain Sustain、timbre brightness 0.7、inharmonic 0、unison 1。主検査は `Consonance` target のPeak/Density、spacing 0 と0.25 ERB、tension 0 と0.4。出生前環境は440 Hz持続Sineの共有解析を72 hop進めたものとし、採点時のframeを64、支持時刻をframe終端以内とする。zero fallbackは候補massを試験注入境界で0へ固定した別条件。実子の連続jitterについて、選択bin中心と最終周波数bitの不一致を少なくとも一例確認する。fixture値を変える場合は取得前に改訂を残す。

`on_spawn_action` の試験専用入口で候補表を受け、現行 `resolve_strategy_frequency` のbin選択とjitterを通す。通常分岐に別の分布器を置かない。注入表は対象 `Field`、population、ID列、member index、spawn counter、frame、解析epoch・支持時刻、range内の全bin slotと実子Recipeを照合する。失効・欠測・unsupportedは選択前に明示拒否し、0 scoreや点地形へ黙って戻さない。失敗時のcounterとID消費を記録する。

検査条件: PeakとDensityの両方で全binを評価する。Densityでは `rho` と身体の変更でmass・抽選確率が変わり、同じ `fitness_score` でもmassが違う対照を置く。独立 `WeightedIndex` と選択後RNG probeを照合する。spacingあり・全占有・全massゼロ・狭いclamp範囲・解析域外・Uniformでは従来fallbackを確認する。連続jitterがbin中心から動くseedを含め、選択前の中心Recipeと実子の最終Recipeを取り違えない。実子のID、世代、周波数bit、body、Recipe hashと、runtime eventを照合する。通常分岐は同じseedの既定OFF対照で変化しない。対象focused testのみ実行し、full suiteとtimingは別工程へ残す。

## 焦点試験の取得状態

2026-09-27、上記の取得条件を固定してから、Consonance のPeak/Density全候補と連続jitter、0 massと全占有のfallback、合成2binにおけるrho依存massの算術と候補mass・fitness levelの不一致、解析epoch・有効level・space・template・ID・候補周波数の破損拒否を5件のfocused testで検査し、5件通過した。抽選binは独立 `WeightedIndex` と照合し、選択後RNG probeも対照と一致。`cargo check --lib` と `cargo clippy --lib -- -D warnings` は通過。`frequency.rs` の狭い範囲・解析域外・Uniform等の従来試験は、F4f身体注入を通さない別の検査である。`cargo clippy --tests -- -D warnings` はF4f以外の21件で失敗。full suiteとtimingはここでは未取得。取得と未取得の詳細は[結果記録](body-aware-fitness-f4f-initial-results-20260927.md)に分ける。
