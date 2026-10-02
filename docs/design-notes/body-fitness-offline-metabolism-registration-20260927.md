# 第十三版: 通常renderの身体代謝入力（取得前登録）

状態: 実装・取得前。第十二版bを保存した隔離worktree `body-fitness-metabolism` で、F4aの一更新検査を通常のシナリオ・Voice・代謝・report経路へ接続する。新設定 `body_fitness_metabolism_offline` は既定falseで、決定的な `conchordal-render` 配線だけが受理する。実時間instrumentは起動時に拒否し、重い代表発音計算を音声hopへ導入しない。これはF4の通常render接続であり、非同期実時間代謝、出生評価、長期生態の完成ではない。

## 入力と時刻

移動用候補表やtarget基音を代謝へ転用しない。各lifecycle更新のautonomous/glide後の実基音と、その時点の実BodySnapshotから周波数代表Recipeを作り、72 hopの実Tone/NSGT密度をF2評価器へ渡す。Gated onsetも発声gateと分離し、実onsetへrechargeを適用する直前の現在基音で同じ評価を行う。基礎費用・連続recharge・内部attack・外部onset・LifeAccumulatorの既存更新順序を維持する。gate、予測、移動はこのflagで変更しない。

環境は前hopまでの受理済み実PCM自己除去結果へ当該hopのhabituationを適用したfield score。source/世代/出生、epoch、Log2Space、時刻を照合する。全substepを含め支持ageが4800 samples以内となるよう、hop終端までの鮮度を検査する。frame 0だけは、全sourceが未発音という初期履歴の共有解析を使用し、専用のoriginとして記録する。それ以降に受理済みの対応source環境が無い場合は、点評価や0へfallbackせずrenderを失敗させる。途中出生・解析設定変更の一般的な再受理は今回の対象外。監査で前hopのsource集合と新Voiceが一致しない経路を確認したため、初期Spawnだけを最大4声・ID一意で受理し、途中Spawn、ReleasePopulation、SetRespawnPolicy、SetHarmonicityParamsは配線前に明示拒否する。

Voiceごとに直近一つの完全Recipe identityと密度だけをcacheし、現在周波数・身体・代表条件が異なれば再計算する。位相0の周波数代表契約を維持する。環境が変わればscore/levelは毎回再評価し、過去のfitnessを流用しない。offline計算の待ち時間とallocationは実時間性能の証拠にしない。

## 固定した最初の検証

通常renderはseed 7、48 kHz/512、0.64秒、Harmonic 440 HzとSine 466 Hz、両bus・amp 0.06。HarmonicはEntrain Sustain、endurance 2秒、recovery 10秒、attack cost/recharge 0、固定音高。SineはDrone・固定音高の環境源。kernel・habituation設定は既定。metabolism OFF/ONを各二回、movement flagはOFF。実Entrain energy、評価score/level、support、現在基音、cache計数を記録する。ON二回のWAVと決定的な代謝/energy記録を照合する。OFFを旧経路の対照とし、frame48のHarmonic energy差が1e-6より大きいことを要求する。clampで差が消える条件は失敗として保持し、取得後に係数を調整しない。

取得前の診断補足: 既存 `population_step` にはenergyが無く、ON専用代謝recordだけではOFFとの差を読めない。両flag状態で同条件に出る `population_step` へ `entrain_count` と `mean_entrain_energy: Option<f32>` を加える。生存Entrainだけを平均し、いないpopulationはnull。Harmonicを一人のpopulationへ置くので、この平均が実Voice energyそのものになる。報告の `time_sec` を48 kHz/512のhopへ換算し、frame48に対応する値を両modeで比較する。新field以外の既存 `population_step` の意味は変えない。OFFとONを各二回取得し、それぞれWAV、同種代謝record（OFFは不在）、population recordの一致を検査する。比較幅1e-6は固定のまま。

内部検査では、F2直接値との一致、休符でも評価可能、currentとtargetの不一致、旧point Cの毒入れが代謝へ漏れないこと、scoreとlevelの両消費、実Gated onset rechargeとgateの分離、source/世代/期限/space拒否を確認する。既存F4a試験を残し、新しい通常providerの経路を別に検査する。設定が実時間配線へ入る負例も必要とする。未取得の条件は結果で明示する。

## 終了条件と残余

focused後にformat・標準Clippy・全target check・全cargo testを実施し、sourceと生記録を固定する。非Sineの実時間移動試験ではこのflagをOFFにして測定条件を混ぜない。移動/生存/全接続の生態除去比較、親/出生の通常runtime接続、欠測時の公平な非同期代謝政策、資源・感度・作者採用は後続の独立条件として残す。
