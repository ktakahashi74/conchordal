# F4a: 実Voice一更新の身体評価と代謝入力

日付: 2026-09-26。状態: 実装・取得前登録。対象は基準`06a4772`と封印済みF3dの19 sourceを起点とする隔離worktree `.worktrees/body-fitness-lifecycle` の`cfg(test)`経路。通常runtimeへの採用、出生、親選択、連続する生存・音色選択の評価は含まない。[F4消費境界の監査](body-aware-fitness-f4-consumer-audit.md)にある成人部分だけを試す。既存`None`経路の既定動作は変えない。

## 入力と一更新の境界

48 kHz、512 sample/hop、通常解析設定、seed 7。source id 1・generation 0のHarmonic 440 Hz実Voiceと、source id 2のSine 466 Hz Toneを構築する。実`ScheduleRenderer`の混合PCMとsource別habitat PCMを64 hop投入し、F3bのsource・世代・epoch・支持終端・受信時刻の検査を通過したsource 1自身を除く一つのLOO環境を使う。qはこのVoiceの固定BodySnapshot、現在基音、source idを使う実Toneの72 hopから求める。F2で同じqと同じLOO有効Cから`BodyFitness { score: S, level: L }`を一回計算する。`S`と`L`は有限、`0 < L < 1`、質量は正、現在基音とtargetの各qから計算したscore差は`1e-5`超を必須とする。成立しない場合はfixture失敗として記録し、取得後に音色・閾値・外部音を選び直さない。

一回の制御substepで実Voiceの現在基音を評価する。試験用のtargetを466 Hz、Glide時定数を0.02秒、`dt=0.01`秒へ固定する。Voice内の`pitch_ctl.force_set_target_pitch_log2`はこの反例の初期状態設定だけに使い、実際の移動判断の成立とは呼ばない。既存commit順序の`update_articulation_autonomous`を一回だけ行った後、lifecycleの前に得た実`body.base_freq_hz()`を評価キーとする。targetに対応するqやF3dの移動候補表を代謝へ流用しない。実行前後にsource id・generation・身体世代、body recipe、current／targetのf32 bit、環境epoch・支持終端・受理時刻、`S`・`L`、point Cの値、energy、内部attack件数、外部onset件数、`LifeAccumulator`の初期・最終値を保存する。制御substepの後にphonation hopを進める場合は、phonation時点の現在基音と評価の対応を再照合し、ピッチ・身体・環境版が変わった評価を流用しない。

接続は既存`Voice::commit_decided_control`を維持し、そのautonomous更新直後、lifecycle呼出し直前に`#[cfg(test)]`の明示offline評価を一箇所置く。試験スレッドだけの具体的な評価contextに解析器、受理LOO環境、Tone recipe、期待source・身体世代・時刻を保持する。評価時の実`&Voice`から現在基音とBodySnapshotを読み、対応するqと`BodyFitness`を一度だけ算出する。contextが無い通常経路は従来の点Cをそのまま使う。contextが有るのに受理・対応検査が失敗した場合はtestを失敗で止め、`None`へ黙って戻さない。本番commitを別の共通helperへ移さず、`Voice`の`Debug`実装用のclosure包装も作らない。lifecycleのlevel/score入力とLifeAccumulatorへ同じ評価を渡す。Gated onsetではphonation gateへ既存point level、`apply_phonation_onset`へその時点の身体`L`を渡す。scoreは連続recharge、levelは基礎費用・attack recharge・onset recharge・LifeAccumulatorに使う。

Entrainの試験初期energyは0.4、enduranceは10秒、recoveryは1秒、attack costは0.05、attack rechargeは0.20、dissonance penaltyは1、rhythm rewardは無効とする。係数・energy clamp・attackの既存更新順序を変えない。各比較を独立した同一初期状態のVoiceから開始し、すべての期待deltaを既存`MetabolismPolicy`の`basal_delta`、`continuous_recharge_delta`、`attack_delta_with_recharge_multiplier`で別計算する。全試験でenergyを0と1の間に保ち、clampで旧入力と身体入力の差が消えた比較は不合格とする。`LifeAccumulator(first_k=1)`を付け、当該一tickの`c_level_firstk_mean`が実際に代謝へ渡した`L`と一致することを確認する。記録欄の名前は既存のままでも、点Cの平均として解釈しない。

## 最小の検査群

1. **休符・level経路と現在基音。** 発音しないhopで内部attackも外部onsetも0件とし、固定身体の代表qと受理LOO環境から`S,L`を計算する。無発音でも身体評価が有効で、basal費用・閾値なしの連続recharge・LifeAccumulatorへ同じ`L`が届く。current pitchとtargetを異ならせ、targetでの評価を誤って使うとenergy期待値に差が出ることを確認する。旧point Cのscore/levelを低値・高値へ毒入れしても、身体経路のenergyとLifeAccumulator値は同じ。旧`None`経路はそれぞれのpoint Cに応じて変わる対照を残す。
2. **score閾値経路。** 同じseedと初期状態の別の参照Voiceをautonomous更新だけ一回進め、その現在基音と封印LOOから事前に`S_ref`を計算する。試験Voiceへ連続rechargeのscore窓`[S_ref-0.3,S_ref-0.1]`と`[S_ref+0.1,S_ref+0.3]`を設定し、実commit中の`S`が`S_ref`とbit一致することを先に確認する。`articulation.process`へ渡すscoreも`S`とbit一致させ、身体scoreによる信号が順に1と0、両energy差が正であることを確かめる。旧selection scoreを毒入れしても結果を変えない。閾値なしの第1群はlevel経路の対照。このscore窓は配送を識別する試験用fixtureであり、生態実験の閾値や作者既定の採用案ではない。窓は先に定めた式で機械的に生成し、実測結果で調整しない。
3. **内部attack。** 既存のtest fixtureと同じくEntrainの`rhythm_phase=2π+0.1`と開いた発火条件を置き、一制御substepで内部attackをちょうど1件起こす。外部phonationは呼ばない。body `L`で計算したattack cost・rechargeが基礎費用・連続rechargeへ一回だけ加わり、`LifeAccumulator`は同じtickを一回だけ数える。attackが0件または2件なら失敗。
4. **実Gated onset。** 内部attackが起きない初期位相から実`Voice::tick_phonation_into`のGated engine・固定clock・`OnsetRule::Always`を通し、受理した一つのonsetとそのstrengthを記録する。`PhonationGate::Immediate`で知覚gateを同じ開状態に固定し、point Cの低値・高値を与える二試行でも発声command・onset時刻・strengthの列が一致するようにする。外部onset件数は各試行ちょうど1件。`apply_phonation_onset`へ届く値だけをbody `L`に分離し、energy差を`strength × (-0.05 + 0.20 L)`を含む既存式と照合する。内部attack分を重複して足さず、直接`apply_phonation_onset`を呼んだだけの試験で代用しない。
5. **gateの分離と拒否。** 別の`WhenViable`実Voiceではbody `S,L`を固定したまま、旧point levelをviability低値の上下へ動かしてonsetの抑止／解放を確かめる。これは発声gateの検査であり、onset列の同一性を要求する第4群と混ぜない。受理LOOのsource・世代・epoch・期限、またはqの現在基音・身体世代が不一致なら身体評価を既知0として消費せず、この明示offline試験を失敗で止める。失敗前にenergy・LifeAccumulator・onsetが変わらないことを確認する。

## 判定と残余

上の5群を最小単位とし、各群の実Tone PCM、受理環境、入力キー、`S,L`、energy式と実値、二種類のattack/onset件数、発声列、LifeAccumulatorを構造化記録に保存する。旧`None`の対照は同じseed・clock・point地形で実行し、基準版の対応する音声・energy記録も保存する。必要な評価が欠けた場合は失敗を保持し、silent fallbackや既知0への置換で合格にしない。全`cargo test`と通常のfmt／Clippyを変更後に実施し、source・binary・入力・結果のhashを保存する。

これは明示barrierの一更新検査。通常runtimeの非同期配送・期限切れfallback、連続Glide中の更新率、queue／総hop資源、出生・親選択、長期生存、作者既定採用は未対応。F3dの一判断成功やF3費用probeの数値を、これらの合格に転用しない。
