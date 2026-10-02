# 通常offlineの反復親付きrespawn: 第十八版bの反復取得結果

2026-09-27。[事前登録](body-fitness-repeated-respawn-registration-20260927.md)のseed 7、founder 3声、死亡率0.5/秒、終了時刻1.72秒を変えずに取得した。第十八版bの全suite内で反復integration試験は成功した。全suite全体も1327成功・0失敗・48 ignore、exit0（2026-09-27 17:32:48 JST）で終了した。[統合検証](body-fitness-repeated-respawn-validation-20260927.md)へ固定物を対応づけた。

## 取得と検査状態

[一次summary](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/repeated-5133-1790497678202911955/summary.json)と[4実行のraw](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/repeated-5133-1790497678202911955/)では、OFF/ON各2回の `conchordal-render` はすべて終了コード0。使用binaryのSHA-256は `3d2c573cfba1c808019d50f8bd2c23573f50a06af7249c8bf5ee15c36c6e36ec`。[取得時のsource manifest](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/repeated-5133-1790497678202911955/source-manifest.json)は414ファイル、SHA-256は `c235a2f40b25799bff59e099b8ff8f6395b3dde68a11ef43904fa548091c89b6`。rawに記録した248ファイルのsource hashはmanifestの対応項目と全件一致した。scene、config、binary、WAV、JSONL、stdout/stderrのhashと実行コマンドは各 `*.acquisition.json` と `*.artifact-hashes.json` に残した。

先行の第十八版focusedでは、[失敗ログ](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/focused-render.log)に終了コード101を記録した。`source_set_after_cleanup` が出生sampleを報告せず、完全identityのassertが停止した。失敗時の[旧raw](/home/shafi/lwrk/conchordal/target/body-fitness-respawn-build-20260927/runtime-respawn-evidence/repeated-867-1790497267094544087/)と[旧peak参照](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/independent-peak-v18/focused-result.json)は保持した。新旧4実行のscene・config hashとWAV hashは一致し、binary hashは修正で変わった。第十八版bではこの報告値とruntimeの完全identity照合を修正し、固定scene・閾値は変更していない。反復integration試験は新しい一次rawで成功した。先行v18の局所respawn試験39件は成功し、v18b全suiteにも含まれる。v18bのformat・標準Clippy・全target checkも終了コード0。

## rawで観測した二機会

死亡は事前RNG-only予測どおり、第一機会がframe `143`・ID `3`、第二機会がframe `154`・ID `2`。ON診断の機会キーはそれぞれ `(opportunity_index, spawn_seq, decision_sample) = (1,1,73216)`、`(2,2,78848)`。OFF・ONともに出生はID `4`、続いてID `5`。両modeで第一poolはID `1,2`、第二poolはID `1,4`。第二poolのID `4` は第一子で、ONの出生前 `source_set` には系譜generation `1`・出生sample `73216` が記録された。第十八版bの `source_set_after_cleanup` もID・系譜generation・出生sampleを全員分記録した。第一cleanup後はID `1,2,3,4`、第二後は `1,2,4,5`。第一子ID `4` は第二cleanup後も `(generation=1, birth_sample=73216)`、第二子ID `5` は `(generation=2, birth_sample=78848)` を維持した。

| 機会 | 親 | ON: 親のhop開始energy → 抽選時energy | OFF: 抽選時energy | 子と系譜 | OFF子Hz | ON子Hz |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 1 | ID 1 | 0.3146169186 → 0.3098237514 | 0.34690475 | ID 4、generation 1 | 448.98578 | 448.3377380 |
| 2 | ID 4 | 0.9550352097 → 0.9505443573 | 0.9505992 | ID 5、generation 2 | 519.48627 | 400.0007629 |

ON第二poolのもう一方のID `1` はhop開始 `0.2619338036`、抽選時 `0.2571463585`。第二親ID `4` の `0.9505443573` は第一出生時energy `1.0` でも第一翌hopenergy `0.9955024719` でもなく、第二機会時点の更新後値として報告された。両機会の実子は予定子とID、member、親、系譜generation、Harmonic body、最終Hz、出生時body generation `0` とRecipe identityで一致する内容をrawが記録した。選択親が第一子となり、第二子の系譜generationが2になったことは今回の観測値であり、登録時に選択親や第二子Hzを固定したものではない。

ONの各機会は全44 bin、16 peak、局所21点を記録し、第一機会の最終身体score/levelは `0.6422325373 / 0.7832088470`、第二は `0.0902886912 / 0.5450220704`。身体scoreと点scoreの差は第一機会のbin295で約 `0.355123`、第二のbin275で約 `0.139791`。別Python実装による[第十八版bのpeak参照結果](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/independent-peak-v18/final-result.json)はON A/Bの二機会、計4 recordの16 bin列にすべて一致した。[実行ログ](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/independent-peak-v18/final-run.log)と[終了コード0](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/independent-peak-v18/final-status.json)を保存した。この参照はrawの身体scoreを入力とし、音響scoreそのものの独立測定ではない。

第一子ID `4` は出生hopのreceiptなし・自己PCM 512 samples全ゼロ、frame `144` のSourceRemoved receiptでenergy `1.0 → 0.9955024719`。第二子ID `5` もframe `154` ではreceiptなし・自己PCM全ゼロ、frame `155` のSourceRemoved receiptで `1.0 → 0.9952416420`。第二機会の出生・翌hop遷移には第一子ID `4` の `(generation=1, birth_sample=73216)` が残り、第二子ID `5` は `(generation=2, birth_sample=78848)` を持つ。出生時のobserver入力batchは各機会3 source、翌hopは4 source。ONのFinish診断はframe `162`・sample `82944`、機会数2と第二子の翌hop受領完了を記録した。

同mode内のWAV hashは2回で一致した。OFFは `131f64218d99c78e1c23bc8bd0bad066fcf58ce6e1269136e5be287470f69b9b`、ONは `81ef0387c05e5b664ea327a043a2404945771511e25d5ad9d87e817c0226eab5`。補助時刻等を含むJSONL全体のbyte hashは同modeでも一致しない。第十八版bの反復integrationでは、WAVと対象JSON recordの同mode一致、および二機会の完全identity照合を含むassertを通過した。先行focusedの失敗と修正後の成功は別の取得として保存した。OFF/ON差は身体出生と代謝を同時に切り替えた結果で、効果を単独に分解できない。固定sceneの二機会、Harmonic body、offline renderer以外の長期生態、実時間性能、他policy、同ID再利用までは検証していない。
