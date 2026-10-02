# F4b: Random respawn一機会の隔離結果

日付: 2026-09-27。登録は[修正登録](body-aware-fitness-f4b-birth-registration.md)。封印結果は`target/body-fitness-f4b-capsule-20260927/`。基準commitは`06a4772c43d06b41b44753bb0be891f23e16b93e`、F3d19 sourceとF4a補足sourceを重ねた隔離worktreeでの試験であり、通常runtimeの身体出生採用ではない。

## 修正前の無効取得

初回targeted試験はpassしたが、外部Sine source id 2が予約子id 2と衝突し、旧Toneは出生時まだhold中でrelease tailではなかった。さらに共有`AnalysisStream::process`の結果へ`recompute_consonance`を呼ばず、全16候補のscoreが0、levelが0.5になった。初回結果・PCM・登録v1は`target/body-fitness-f4b-preliminary-invalid-20260926/`に保存した。初回source全体のhashは取得時に封印していないため、この無効取得の再現性は保証しない。これを成功証拠に使わない。修正登録はこの診断後の登録であり、完全な取得前登録とは呼ばない。

## 実経路の一機会

seed 7、48 kHz、512 sample/hop、frame 64で、死亡Voice id 1のHarmonic 440 Hz Toneと外部source id 3のSine 466 Hz Toneから出生前共有環境を作った。旧Toneはsample 24000で明示Offし、birth sample 32768でもrelease中。旧source単独PCMのRMSはrelease早期hopで0.233326、出生直前hopで0.137309、いずれも正。共有C score scanの範囲は`[-0.671309, 1.0]`、同じ音声から得た成人LOO Cとの差のL1和は41.4552。候補に死者自身の音を除いた成人LOOを渡していない。

実`Community::respawn_on_new_deaths`のRandom一機会で、既存RNGから220–880 HzのUniform候補16 slotを生成。各候補は予約子id 2、population id 7、member index 1、generation 0、parentなし、同じtemplateと出生frameから仮Voiceを実生成し、そのBodySnapshot・代表modulatorを用いる実Tone 72 hopの密度とF2 score/levelを得た。候補score範囲は`[-0.159027, 0.387682]`、正scoreは7/16。重み付き選択の独立再計算は選択周波数bitと選択後RNG probeが実経路と一致した。候補はslot順で保持し、周波数重複が生じてもslot数と重みを統合しない。この固定seedでは重複なし。

点Cをscore -10/level 0.05とscore +10/level 0.95に変更しても、身体方式の候補、16組のscore/level、選択子、RNG probeは同じ。閾値なしは両方出生。事前式`min(L_i)/2 = 0.210575`では出生、`(1+max(L_i))/2 = 0.842340`では拒否。出生時はspawn counter 1→2、runtime id 2→3、member index 1→2、respawn event 1件。拒否時はcounterだけ1→2、id 2とmember index 1を維持し、eventなし。同じ死亡の再処理ではcounter増分なし。選ばれた子の周波数bit、id、generation、BodySnapshot、代表modulatorは採点時の候補と一致し、これらを含む宣言代表recipe hashも一致した。ただしhold 48000 sample、ADSR(attack .005, decay 0, sustain 1, release .5)秒、tau 0、fs 48000は試験で固定した代表励振条件であり、実子の通常ToneSpecから再取得した欄ではない。通常発音の全recipe一致をこの検査から主張しない。

旧`None`点C方式は固定閾値0.5で低Cを拒否し、高Cで出生した。外部source id 3はこの一機会fixture内で予約子id 2と分離するための値であり、将来のid 3以降の予約一般を保証しない。source集合、live templateのbrightness、予約id、出生frame、epoch、候補keyを個別に壊した6試験はid割当・spawn前に失敗した。templateの検査は各候補でlive templateから子を再構成し、身体とrecipe identityを照合する固定Harmonic試験であり、一般templateの全制御状態に対する失効保証ではない。

## 旧None通常経路と検証

同じseed 7の4.5秒respawn scriptとconfigを取得前にhash固定し、封印06a release renderと現版release renderで比較した。WAVはbyte一致、SHA256=`5b91ce7f0cb4c7278a730f4ddd07b8cfb32009cd2b05b6d749d47e79191dbdd5`。初期spawn 1件、respawn 20件、death 21件を含む非timing record 2252件は構造化比較で一致。実時間依存のhop_timing 423件は比較から除外した。比較の入力・binary・report hashと全respawn eventはcapsule内の`base-none-regression/comparison.json`に保存した。

隔離worktreeの全`cargo test -- --nocapture`は`exit=0`（2026-09-27 00:11 JST）。`cargo fmt --all --check`と通常`cargo clippy -- -D warnings`も通過。source tar、全src/Cargo/登録のhash一覧、raw PCM、候補ごとの密度・score/level、負例、test reportをcapsuleに封印した。主実装は`cfg(test)`の採点境界のみであり、既定`None`の通常音声・出生を変更しない。

この結果はF4b Random一機会の機能検査。Hereditary、PeakBiased、複数死亡の競合、Modal/body patternの地形依存、非同期取得、長期の選択差、測定資源、作者既定採用は未検証。
