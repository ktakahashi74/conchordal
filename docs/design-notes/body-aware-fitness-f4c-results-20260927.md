# F4c: Hereditary一機会の隔離結果と未成立条件

日付: 2026-09-27。状態: 取得登録全体は不合格。実装は隔離worktreeの`cfg(test)`経路に限定し、mainや通常runtimeに接続しない。[登録](body-aware-fitness-f4c-hereditary-registration.md)と`target/body-fitness-f4c-capsule-20260927/`を参照。F4bの封印capsuleは変更していない。

初回targeted試行は、fixtureが`retrigger=false`かつEntrain state Idleだったため、energyが正でも二親の`Voice::is_alive()`がfalseとなり、親poolなしで失敗した。元のstdout全量は次回targeted実行で上書きされたため封印できていない。確認できた失敗内容と修正箇所はcapsule内の`first-targeted-failure-note.md`に残した。登録にある生存Entrain二体を作るため、`retrigger=true`、`autonomous_attack=false`に修正した。seed、音源、候補数、sigma、閾値式は変更していない。

修正後の一機会では、旧id 2のToneをsample 24000でOffし、birth sample 32768にrelease tailを保持した。旧音のhop RMSは0.233366から0.133165へ低下し、出生直前も正。4 sourceの実ScheduleRenderer PCMをF3b Workerへ渡し、親id 1と3それぞれのLOO環境を取得した。子用共有C score scan範囲は`[-0.282679, 0.997306]`、親LOOとのL1差は順に25.1858、29.6883。旧個体や親を子の自己音として除いていない。

親id 1のF2 score/levelは`0.214379/0.605577`、親id 3は`0.198765/0.598094`。両値はF4a test contextから実`commit_decided_control`へ配送された。親poolは生存id 1/3の二体だけで、energyは0.35と0.65の正で異なる値。既存energy比例抽選と同じRNGから親id 1を選び、sigma 0.03のHereditary候補16 slotを生成。子用の代表Tone72 hop身体level範囲は`[0.494659, 0.759282]`。`max_by`が身体levelを読んで子周波数bit `1134839946`を選び、選択後RNG probeは独立再演と一致。点Cの低・高毒入れで親評価値、親抽選、候補、子、終了RNGは一致した。固定式の低閾値0.247330では出生、高閾値0.879641では拒否。出生時の子id 5、generation 1、親id 1、member index 3、BodySnapshot、宣言代表recipeのbody/freq/modulatorは採点候補と一致。拒否時はcounterだけ進み、idとmember indexは不変。

ただし、**親の実energyは更新前後で双方bit不変**。このSustain fixtureはendurance/recharge係数が0であり、身体score/levelを配送してもenergyへの作用が現れなかった。したがって「F2→実energy変化→親選択」の因果鎖は未実証で、F4c全体を合格にしない。抽選は初期値由来の正で相異なるenergyを使った。結果に合わせてenduranceやrechargeを追加しなかった。登録した失効負例群と試験専用旧`None`点C対照も未取得。宣言代表recipeのhold/ADSR/tau/fsはfixture固定で、実子の通常ToneSpec全欄との一致ではない。

旧`None`通常経路は、seed 7の短いHereditary scriptとconfigを実行前にhash固定し、封印06a release binaryと現版release binaryで比較した。WAV byte一致、SHA256=`635775e739f2021ff7c0a01357d90b488d534879e6f06c9d262014eb6c23f9b0`。respawn 40件、death 43件、初期spawn 3件を含む非timing記録2319件は構造化比較で一致。hop_timing 423件は実時間依存として除外。全`cargo test -- --nocapture`は`exit=0`（2026-09-27 00:29 JST）、fmtと通常Clippyも通過した。これらは実装健全性と旧経路回帰の確認であり、未成立の代謝因果条件を補わない。

次回は非ゼロ代謝係数を取得前に明示し、親抽選前のenergy変化を合格条件に置く登録が必要。現試験の結果を見て係数を追加する処置は行わない。
