# 通常offline反復respawnの統合検証（第十八版b）

2026-09-27。隔離worktree `.worktrees/body-fitness-repeated-respawn` で、通常offlineの二度目の親付きrespawn、第一子を含む親pool、二度の出生から自己除去観測への遷移を検証する。[事前登録](body-fitness-repeated-respawn-registration-20260927.md)、[数値結果](body-fitness-repeated-respawn-results-20260927.md)、[検査範囲](body-fitness-repeated-respawn-focused-review-20260927.md)を分けて記録する。

## 固定物と変更範囲

第十七版fの414 sourceをhash照合して引き継いだ。変更は `src/life/community/respawn.rs`、同 `respawn/offline_respawn.rs`、同 `offline_respawn_tests.rs`、`src/runtime/mod.rs`、`tests/body_fitness_respawn.rs` の5ファイル。最終sourceも414ファイルで、[manifest](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/integration-source-20260927-v18b/manifest.json)のSHA-256は `c235a2f40b25799bff59e099b8ff8f6395b3dde68a11ef43904fa548091c89b6`。[source archive](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/integration-source-20260927-v18b/source.tar.gz)を保存した。

評価器の永久的な一回限りの状態を、決定sample、死亡sourceの完全identity、spawn sequence、機会連番へ変更した。未排出recordやpending、同sample・同死者・sequence逆行は拒否し、新機会は親・予定子・候補scratchを更新する。実装に「ちょうど二回」の上限や登録frameの特例は加えない。source上限4、単独死亡、親2声、静的epoch、既定OFF・offline専用という境界は維持する。

runtimeは初期founder以外のsourceも出生sampleを保持し、退役許可を完全identityで追跡する。前児の翌hop receiptはcleanup前に検証し、遷移reportはcleanup後の集合で出す。出生・翌hop・自己PCMには機会連番とspawn sequence、死亡sourceのidentityを付ける。第一子の出生sampleを第二機会で0へ戻さない。親の音色を遺伝する変更ではなく、子の身体は固定Population templateに基づく。

## 事前条件と取得

seed7、Harmonic Entrain Sustain founder3声、380–520 Hz、48 kHz/512、背景死亡率0.5/秒を維持し、Finishだけ1.72秒へ延ばした。RNG-onlyでは死亡をframe143/ID3、154/ID2、225/ID5と予測した。通常sceneの生存条件は音声取得で確認する対象であり、この事前予測だけでは証明しない。OFFは身体代謝・respawnをともに無効、ONはともに有効で各二回。処置ごとの効果を分離する比較ではない。

最終全suiteは1327成功・0失敗・48 ignore、`cargo test exit=0 @ 2026-09-27T17:32:48+09:00`。[test_report.txt](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/test_report.txt)にstdout/stderr、[test_status.txt](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/test_status.txt)に同shellの実終了コードを保存した。fmt・標準Clippy・全target checkもexit0。先行v18のfocused lib39件も最終suiteに含まれる。

新二機会の通常render4回と旧一機会・高閾値拒否の5回はすべてexit0、通常scenario拒否13条件も通過し、respawn integration3件すべて成功した。最終保存binary SHA-256は `3d2c573cfba1c808019d50f8bd2c23573f50a06af7249c8bf5ee15c36c6e36ec`。最終二機会rawのON A/B、計4機会のpeak候補列を別Python実装で再抽出し、すべて一致した。取得ごとのsource snapshot248ファイルと固定manifest414ファイルの対応も照合した。

[機械可読対応表](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/validation.json)と[一次取得索引](/home/shafi/lwrk/conchordal/.worktrees/body-fitness-repeated-respawn/target/repeated-respawn-validation/primary-acquisition.json)へsource、binary、実行command、scene/config、raw、検証ログ、関連文書を対応づける。build directoryは第十七版と共用するが、各版のbinaryは個別保存し、rawの固有directoryを上書きしない。

## 初回失敗と修正

v18のfmt・Clippy・全target check、focused lib39件は通過した。通常render4回もすべてexit0だったが、integrationは `source_set_after_cleanup.birth_sample` の欠落でexit101となった。前後のsource identityを照合する試験を緩めず、v18bでcleanup後のreportにも出生sampleを追加し、runtimeが期待する完全identityと照合するようにした。scene、seed、係数、閾値と期待source集合は変更していない。

失敗source・binary・logは `target/integration-source-20260927-v18/`、先行rawは `runtime-respawn-evidence/repeated-867-1790497267094544087/` に保存した。そこでのpeak独立照合4/4成功は、integration成功とは別の結果として保持する。修正前のbody scoreや音声の数値を、修正後の全suite成功へ置き換えない。

## 独立対照と残余

第十七版で残ったpeak抽出は、[別言語の参照検査](body-fitness-respawn-peak-reference-20260927.md)で全binから再計算した。候補scoreは記録値を共有するため、音響scoreの独立再測定ではない。Python倍精度とRustのf32の差があるため、閾値近傍の全入力へ等価性を拡張しない。別環境fixtureによるTone直接積分、通常renderの実消費、production関数への局所fault注入も別の証拠である。

局所試験は、前児を次の親poolへ入れた二度の実cleanup、候補identityの全置換、source追加/不正削除/活動中ID衝突/系譜、出生sample、自己PCM欠落/511samples/非ゼロ、翌hop子・receiptの欠落を対象とする。runtime recordの機会キー混線とnext receiptの古いdecision clockを直接注入した試験はない。登録した全faultを独立検証済みとは扱わない。

同じhopで前児の翌hop確認と次の機会が重なる場合、現版はcleanup後のrecord処理で拒否する。前児receipt検証後でも拒否し、RNG・counter・子ID消費の巻戻しは保証しない。この制限はframe143→154の二機会とは別の未解決事項であり、一般の反復生態の完了とは呼ばない。

次は[全消費者比較の設計草案](body-fitness-all-consumers-draft-20260927.md)に従い、F4の本来の比較へ進むため、同hop重複の処理、固定異種身体、移動のみ／生存のみ／全接続の判断時刻と有効評価率を整える。今回のsourceで実時間専有取得やrelease build、実device、試聴・作者受入は行っていない。音色遺伝、同ID再利用、F4/F5/F6全体は未完了。mainのsrc・既定設定・A4基準版は変更せず、commit・pushは行っていない。
