# F4b: Random respawn一機会の身体評価登録

日付: 2026-09-26。改訂: 2026-09-27。状態: 初回targeted取得後の修正登録。初回版は`target/body-fitness-f4b-preliminary-invalid-20260926/registration-v1.md`に保存した。初回取得は外部source id衝突、持続音をtailと呼んだ誤り、共有解析後の`recompute_consonance`欠落により、16候補すべてscore 0・level 0.5の無効結果。以下はその診断後に固定した再取得条件であり、完全な取得前登録とは呼ばない。対象は隔離worktree `.worktrees/body-fitness-lifecycle` の `cfg(test)` 明示offline経路。F4a補足capsuleは固定済みで変更しない。通常runtime、作者既定、Hereditary、PeakBiased、補助settle、非同期、長期選択には適用しない。

## 固定入力と因果境界

Community seed 7、48 kHz、出生機会frame 64、512 samples/frame、population id 7、死亡Voice id 1、初期member index 1、予約子id 2、出生世代0。templateは基音440 Hz・固定ratio Harmonic・Entrain Sustainで、身体patternの地形依存やModal位相を使わない。旧Voiceは固定440 Hzへ初期配置し、試験用Population stateのrespawn候補strategyだけ`SpawnStrategy::Field`のUniform/Density、220–880 Hz、最小ERB距離0へ固定する。`RespawnPolicy::Random`、capacity 1、`respawn_settle_strategy=None`。16候補の生成と選択は既存Community RNGをそのまま使用する。`spawn_counter`は機会ごとに1進む。拒否ではidとmember indexは進まず、成功時だけ各1進む。死亡は既存`force_dead`相当の状態遷移で作り、同じ死亡を二度処理して二度出生させない。

出生前共有環境は48 kHz・512 sample/hopの実ScheduleRendererで、id 1のHarmonic 440 Hz Toneと外部id 3のSine 466 Hz Toneを64 hop混合したPCMから実AnalysisStreamで構築し、`recompute_consonance`を適用する。外部id 3はfixture内の別sourceであり、予約子id 2と衝突しない。旧Toneは時刻0開始、hold 24000 sample、時刻24000に明示Off、振幅0.35、attack 0.005秒、release 0.5秒。時刻32768 sampleで旧Voiceのhabitat PCMが正かつ減衰中であることを複数hopの包絡指標で確認し、死を観測した後もその尾を混合PCMに含める。出生時に死者自身を除く成人LOOへ置き換えない。共有環境はepoch 31、support end/received/decisionすべて32768 sample、出生候補すべて同じ版。実測PCMと共有C scan、旧Voiceを除いたLOO C scanの差、共有score scanが非定数であることを保存する。環境の期限は試験用に4800 sample以下、future/missing/staleは失敗とする。

候補slotごとに同一template、候補周波数、予約id 2、出生frame 64、metadata(population 7、member index 1、generation 0、parent None)、Community seed 7、出生地形から`VoiceSpec::spawn_with_landscape`で仮Voiceを生成する。仮生成に候補選択RNGを渡さない。仮VoiceのBodySnapshotと実Tone代表recipeから72 hopの主観強度密度を測り、出生前共有環境の有効CとF2式でscore/levelを計算する。unsupported、無帯域質量は失敗。同一周波数が複数slotに現れても各slotと重みを保持し、選択確率を統合しない。選択後の実子についてid・generation・周波数bit・BodySnapshot・recipe hashが採点時と一致することを必須とする。

## 既存選択則と閾値

既存16候補を同じ順に採点する。scoreの正部分を`WeightedIndex`へ渡し、全重みゼロなら最大scoreを選ぶ。候補生成後のRNGをcloneした独立参照で選択値と選択後RNG fingerprintを照合する。点Cを採点へ混ぜない。閾値なし、全候補levelより低い閾値 `min(level_i)/2`、全候補より高い閾値 `(1+max(level_i))/2` の三独立初期状態を試す。閾値は全16候補の採点完了後に式で確定し、選ばれた候補で調整しない。低閾値は成功、高閾値は拒否。旧点Cだけをscore -10/level 0.05 と score +10/level 0.95 に変える毒入れ対照で、身体方式の16候補・score/level・選択・閾値・RNG・出生は同一。`None`方式は既存の点C依存を対照として記録する。

成功時は出生実経路のruntime event、子id、member index、body、recipe、周波数、parent None、spawn counter、next runtime id、death-observed集合を保存する。拒否時は子/eventなし、counterのみ進む。source/template/id/frame/epoch/候補keyが一致しない場合はid割当・spawn前に明示offline試験を失敗させ、ゼロscoreや`None`へfallbackしない。失敗時のcounter消費は既存の機会開始済み一回を許容する。選択関数自体を複製せず、その採点入力のみ`cfg(test)`境界から置換する。

旧`None`通常経路は同一seed・respawn scenarioを基準`06a4772`封印renderと変更版release renderで実行し、WAVとenergy・出生eventを含む非timing主要recordを比較する。scenarioとconfigは実行前hashを保存する。test専用hookがrelease音声・出生を変えないことの確認であり、身体方式の通常runtime採用を示さない。全`cargo test -- --nocapture`、fmt、通常Clippyを変更後に実施し、source/input/binary/result hashesを別capsuleへ保存する。

この一機会は成人代謝から親energyが形成される長期過程、複数死亡の順序、ModePatternの地形依存、Modal位相、欠測時の演奏継続、評価率、作者既定を検証しない。
