# 第十四版: 通常offline初回Field出生の身体評価（取得前登録）

状態: 実装・取得前。v13b405 sourceを複写した `body-fitness-recovery` worktreeで、F4fの試験専用注入に留まった初回Field出生を、決定的な `conchordal-render` の通常Scenario/Conductor/Community経路へ接続する。独立設定 `body_fitness_birth_offline` は既定false。実時間instrumentは音声デバイス初期化前に拒否する。評価は出生機会で同期実行し、音声hopの一般経路や実時間性能を保証しない。

取得後注記: 下記の最初のConsonance Peakシーンはspacingを明示せず、Rhai既定1.0 ERBが適用された。44候補中、環境440 Hzから1.0 ERB以上離れた候補はbin 311だけ。テスト側の「全候補level最大」判定は既存の非占有優先規則を見落とし、初回取得は失敗として保持する。生記録 `target/body-fitness-recovery-build-20260927/runtime-birth-evidence/normal-72-1790487502174761377/`、ログ `target/body-fitness-recovery-normal-first.log`。取得済み条件や結果を後から書き換えず、次の[第十四版v2登録](body-fitness-offline-birth-v2-registration-20260927.md)へ分ける。

## 経路と制約

WorkerStateが一つのoffline出生評価器を所有し、Conductorのaction dispatch時にだけCommunityへ可変参照を渡す。thread-local注入をproduction化しない。出生前の `current_landscape` は前hopまでの受理済み共有PCM解析に当該hopのhabituationを適用したもの。初回環境音源はtime 0に出生し、約64hop後にField子を一声出生させる。環境support時刻、受理epoch、Log2Space、habituationの有効scanを出生評価時に照合し、欠測・古さ・不一致は点評価へ黙って戻さない。出生は全sourceの自己除去環境ではなく、出生前の共有環境を用いる。途中Spawn用のsource-removed observerはこのflag単独では有効化しない。

対象は初回 `SpawnStrategy::Field` のConsonance/Dissonance/Edge、Peak/Density。Gap/UniformとField以外は旧経路を使い、対象外として診断する。予約された子ID・member index・世代0・現frame・Community seed・実template・各候補binのclamp済み周波数から仮の子Voiceを構築し、位相0の代表Recipeで72hopの実Tone/NSGT密度を作る。F2の身体平均fitness score/levelと、Consonance/Dissonance/Edge各々の身体平均density massを算出する。Peakではlevelまたは0.5距離、Consonance tensionではscore範囲からの目標距離を用いる。Densityではmassにtension係数を掛け、既存の占有判定、zero fallback、`WeightedIndex`、bin内log jitterをそのまま通す。bin中心の選択後、最終jitter周波数で別の実子Recipe/fitnessを評価し、実出生VoiceのID・member・body・Recipe identity・runtime eventと照合する。

同じ `body_fitness_birth_offline` と `body_fitness_metabolism_offline` の同時ONは今回拒否する。既存の `body_fitness_action` または `body_fitness_observation` との同時ONも初版では明示拒否し、出生前共有環境と既存source別観測の意味を混同しない。通常の既定OFF経路は変更しない。実時間instrumentのflag拒否はrender開始前に行う。出生対象に後続の親付きrespawn、解析パラメータ変更、release、同一人口への再spawnは含めず、初版のScenario検証で拒否する。4 Voiceを超える条件も拒否する。

## 固定した取得条件

48 kHz/512、seed 7、既定kernelとhabituation設定。time 0にSine 440 Hz・Drone Sustain・両bus amp 0.06を一声置き、`wait(0.6826667)` の後にHarmonic Entrain Sustain・brightness 0.7・inharmonic 0・unison 1の子一声を380–520 HzのFieldで置く。出生後 `wait(0.16)` で終了する。Consonance Peakを主例、Consonance Density、Dissonance/EdgeのPeak/Densityを内部の独立F2対照とする。flag OFF/ON各二回のWAV、出生event、身体出生診断を保存する。ON二回の同種recordとWAVは一致を要求する。OFF二回も同様に一致。ONの選択binは全候補F2 levelのPeak最大と一致し、実子最終Recipe identityは診断と一致する。少なくとも1候補で身体levelと旧fieldのpoint levelに `1e-6` より大きい差を要求する。ON/OFFの子周波数差だけでは身体因果を主張せず、候補fitnessと旧field値の非同値、独立選択を合わせて判定する。診断には対象target、sampling、子ID/member、候補数、選択bin・中心Hz、最終Hz、最終score/level/mass、Recipe hash、環境epoch/support/decision sample、fallback種別を記録する。

負例はflag OFFの旧経路再現、instrumentのheadless `--play=false` 起動拒否、metabolismとの同時ON拒否、action/observationとの同時ON拒否、欠測/古いsupport・epoch/space不一致、対象外targetの旧経路保持。取得後の閾値緩和、seed/身体/出生時刻の変更は結果へ旧失敗と改訂登録を残す場合だけ行う。focused testだけを実行し、全suiteとtimingは統括作業へ渡す。
