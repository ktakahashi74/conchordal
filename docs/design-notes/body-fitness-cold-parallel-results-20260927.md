# 第十五版b: 冷候補準備の有界並行化・取得結果（2026-09-27）

[取得前登録](body-fitness-cold-parallel-registration-20260927.md)の条件で、固定した第十五版bの source と release test binary を用い、音声デバイスを使わない paced runtime を専有取得した。対象 source manifest の SHA-256 は `30d7d1b19289799a767bd02366d04556b5138779bcc5cedb1b37765d77cf9e43`、両取得で使用した binary の SHA-256 は `efb13e36c5ba24f8739c5c53a964447d1db696fb45c7c0e2f02ad753f0285224`。source、binary、実行コマンドは[非Sine取得記録](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-non-sine-v15b-20260927/acquisition.json)と[Sine取得記録](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-sine-four-v15b-20260927/acquisition.json)、各[非Sine raw manifest](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-non-sine-v15b-20260927/raw/manifest.json)・[Sine raw manifest](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-sine-four-v15b-20260927/raw/manifest.json)に残した。

## 登録条件と合否

非Sine条件は seed 17、Harmonic 220 Hz と Modal 330 Hz の2声、10秒間の主観測窓938 hop、frame 300のbrightness変更、frame 600のcontrol変更で、action OFF／ONを各1回取得した。Sine条件は1声・4声それぞれのOFF／ONを同じ10秒窓で取得した。6条件すべて `pass: true`。主観測窓では指定声数を維持し、callback相当のunderflowとhop予算超過はすべて0。support ageとcache上限も全条件で妥当と判定された。hop予算は10.667 ms。

| 条件 | action | 最大hop時間 | 消費判断 |
| --- | --- | ---: | ---: |
| Harmonic＋Modal 2声 | OFF | 1.258 ms | 対象外 |
| Harmonic＋Modal 2声 | ON | 1.211 ms | 117（Harmonic 62、Modal 55） |
| Sine 1声 | OFF | 0.648 ms | 対象外 |
| Sine 1声 | ON | 0.986 ms | 97 |
| Sine 4声 | OFF | 2.180 ms | 対象外 |
| Sine 4声 | ON | 2.348105 ms | 599（152、155、143、149） |

消費判断の件数は最後の定期report（frame 912）の累計であり、938 hop全窓の最終値ではない。詳細値は[非Sine summary](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-non-sine-v15b-20260927/summary.json)と[Sine summary](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-sine-four-v15b-20260927/summary.json)を参照。ON条件のobserverは停止せず、各912 batchを受理した。cache判定の256 entry／1 MiBはsourceごとの密度cacheに対する上限であり、プロセス全体のメモリ上限を測った値ではない。実装上の最大8計算threadは4 source workerと各補助laneの合計であり、解析workerなど他のthreadを含まない。

## 非Sineの冷準備と変更後の回復

ONの最初の冷準備はHarmonicが133候補・980,743 µs、Modalが135候補・990,588 µs。いずれも `compute_lanes: 2`、全候補完了、Readyとして記録された。これは当該jobの計算区間であり、別世代jobや単独1lane実行に対する速度向上倍率ではない。候補数と非同期の決定時刻は旧版と動的に異なるため、旧版との単純な倍率比較も行わない。

frame 300（sample 153600）の身体変更で、Harmonicの旧世代serial 20に取消を要求し、frame 301（sample 154112）で取消応答を受けた。このjobはwarm逐次経路の `compute_lanes: 1` で、133候補中10候補を完了していた。取消済み結果を判断に導入していない。診断は[非Sine ON raw report](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-non-sine-v15b-20260927/raw/non-sine-changes-2-source-on/report.jsonl)の `last_cancelled_preparation` と[summary](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/runtime-non-sine-v15b-20260927/summary.json)に保持される。

frame 384の定期reportではHarmonicの累積消費は10回で、旧世代の最後の消費が残り、新世代serial 23はpendingだった。frame 432では直近消費の身体世代が2に更新され、旧世代分を含む累積消費が12回となり、登録期限内の回復を確認した。frame 432で見える直近消費serial 31の判断時刻はsample 202752（frame 396）だが、その前の新世代serial 23の完了診断は定期reportで上書きされている。したがって、最初の新世代消費をframe 396と断定しない。取得記録から言える範囲はframe 384より後、frame 396まで。新世代冷jobの実時間と `compute_lanes` も、このreportだけからは確定できない。

frame 288／432／576／720／912のHarmonic累積消費は10／12／21／40／62、Modalは8／20／25／38／55。Modalの身体世代は1を維持した。frame 600のcontrol変更後は `Consume(ControlChanged)` の拒否診断を出し、frame 720までに回復し、frame 912まで消費を継続した。最終action報告は完了118件、消費117件、取消要求1件、取消job1件、拒否1件。

## 検証の境界

固定sourceに対するformat、標準Clippy、全target check、全体test、release test binary buildはすべてexit 0。全体testは1303 passed、0 failed、48 ignored。[検査記録](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/cold-parallel-validation/checks.json)と[件数記録](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/cold-parallel-validation/test-totals.json)を参照。最初の第十五版sourceでは非test compile時のimport条件に不備があり、Clippyが失敗した。その[初回記録](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-cold-parallel/target/cold-parallel-validation/initial-v15/checks.json)を保持し、修正後の第十五版bで上記の検査と取得を行った。

取得後、Sineの要約器が非Sine専用キー `non_sine_changes_checks` を参照して `KeyError` となった。修正対象は後処理だけで、既存rawを再集計した。source、binary、raw取得、機能test、paced取得は再実行していない。

[第十四版の失敗記録](body-fitness-recovery-results-20260927.md)は維持する。第十四版では新世代135候補の計算に1,905,687 µsを要し、最初の消費は期限frame 432を過ぎたframe 480だった。第十五版bの局所合格は、長期生態、実音声デバイス、通常出生と代謝の結合、非同期代謝、音色遺伝、作者による採用を検証した結果ではない。
