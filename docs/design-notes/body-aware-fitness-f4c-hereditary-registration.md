# F4c: 成人energyからHereditary出生一機会までの隔離登録

日付: 2026-09-27。状態: 実装・取得前登録。最初の登録SHAは`d40989e9d69cde14ef6e533943e61150c6d9196d4b8519f0b0007e60c90d1e57`、`target/body-fitness-f4c-prereg-amendment-20260927/registration-v1.md`へ保存した。代表recipeの同一性を数値取得前に明確化した。F4b封印capsuleを変更せず、同じ隔離worktreeの`cfg(test)`経路だけを拡張する。通常runtimeへの採用、音色遺伝、長期選択、PeakBiased、settle候補、非同期出生は範囲外。

## 固定場面

seed 7、48 kHz、512 sample/hop、出生frame 64（sample 32768）、epoch 31。Population id 7へ初期Voice id 1/2/3をLinear 330/440/550 Hzで配置する。id 1と3は生存Entrain親候補、id 2は死亡させる旧個体。templateは固定ratio Harmonic、brightness 0.9、Entrain Sustain。初期energyはid 1が0.35、id 3が0.65。各親の`Voice::commit_decided_control`を一回だけ`dt=0.01`秒で実行し、親ごとのF4a明示offline contextへ、実Toneの72 hop密度と各自だけを除いた受理LOO環境を渡す。内部attack、外部onsetは0件とする。両親の更新後energyが有限・正・相異でない場合はfixture不成立として失敗を保存し、値を取得後に調整しない。親選択はこの更新後energyを既存`weighted_parent_select`で一度だけ読む。身体scoreを親重みに再乗算しない。

出生前の音はCommunityの3体と外部Sine id 4、計4 source。id 1/2/3の実BodySnapshotに対応するHarmonic 330/440/550 Hz Toneと外部Sine 466 Hz Toneを実ScheduleRendererで混合し、source別PCMとともに64 hopをF3b Workerへ投入する。旧id 2のToneはsample 24000で明示Off、release 0.5秒とし、出生時にも正の減衰tailを残す。親id 1/3の更新用環境は、それぞれ自身だけを除いたsource output。子の環境は旧id 2のtailと両親・外部を含む出生前共有解析に`recompute_consonance`を適用したもの。support end/received/decisionは32768 sample。共有Cが非定数、親の各LOOと共有Cが異なることをPCMとscanで保存する。欠測・不一致・期限4800 sample超は0へ置換せず試験失敗。

出生前に外部source id 4までruntime idを予約し、子の未使用idを5へ固定する。親generation 0、子generation 1、子member index 3。初期配置後のrespawn strategyはUniform/Density 220–880 Hz、最小ERB距離0へ試験stateで固定する。`RespawnPolicy::Hereditary { sigma_oct: 0.03 }`、capacity 3、`respawn_settle_strategy=None`。親の抽選後、選ばれた親の基音log2へ既存正規変動を16回加え、既存の220–880 Hz clampを用いる。全候補slotを元順序のまま保持し、同値・重複を統合しない。各候補は選択親idとgenerationを含むmetadata、予約id 5、template、候補基音、frame、Community seed、出生地形から`spawn_with_landscape`で生成する。候補作成に選択RNGを渡さない。実Tone rendererを使う宣言代表励振の72 hop密度と共有CからF2のscore/levelを計算する。代表recipeは候補VoiceのBodySnapshot・基音・代表modulatorを使い、hold 48000 sample、ADSR(attack .005, decay 0, sustain 1, release .5)秒、tau 0、fs 48000を試験で固定する。固定欄を出生後の実Voice ToneSpec由来とは呼ばない。実子との一致検査はbody/freq/modulatorおよびそれらを含む宣言代表recipe hashに限定する。通常発音との全recipe同一性は未検証。Hereditaryの全候補`max_by`へこのlevelを渡し、同値なら既存`Iterator::max_by`と同じ後方slotを選ぶ。採用後の最低level判定も同じlevelを使う。

出生閾値はなし、`min(L_i)/2`、`(1+max(L_i))/2`の三条件。16候補評価後に式から求め、実測の採用候補で調整しない。前二条件で出生、最後で拒否を必須とする。候補score/levelの全値、親poolのid・energy・generation、親選択前後と候補選択後のRNG probe、各親更新前後のenergy・身体評価、候補・子のBodySnapshotとrecipe hash、子parent id/generation・周波数、counter/id/member index/eventを保存する。拒否時はspawn counterだけ一回進み、idとmember indexは不変。同じ死亡の再処理は追加機会なし。

点Cだけをscore -10/level 0.05とscore +10/level 0.95へ変える独立試行でも、親の身体評価・更新後energy、親抽選、16候補、子level選択、終了RNGと出生結果が同じことを検査する。旧`None`には点C依存の対照を残す。親poolに死亡id 2や外部id 4が入らないことを確認し、親抽選の独立再演には同じ候補用RNGのcloneを使う。選択親の世代、live templateのbrightness、子id、frame、epoch、候補slotをそれぞれ壊した負例は子追加前に失敗させる。一般template全制御の失効保証とは呼ばない。

旧`None`の通常release経路はseed 7、固定Harmonic 3体、`.respawn_hereditary(0.03)`、短いenduranceのscriptとconfigを実行前hash固定し、封印06a renderと現版renderを比較する。WAV byteとrespawn/energyを含む非timing主要recordの一致を必須とする。全`cargo test -- --nocapture`、fmt、通常Clippyを実施し、source/input/binary/結果hashをF4bと別capsuleへ保存する。正例条件が成立しなければ失敗と入力を保存し、seed・閾値・音源を調整しない。
