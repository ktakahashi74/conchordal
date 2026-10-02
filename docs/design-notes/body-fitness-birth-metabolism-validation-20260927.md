# 第十六版dの統合検証

隔離worktree `.worktrees/body-fitness-birth-metabolism` に、第十五版bの全409 sourceを内容照合して引き継いだ。通常offlineの初回Field出生と身体代謝の同時接続が対象。変更は `src/life/offline_metabolism.rs`、`src/runtime/mod.rs`、`src/runtime/body_fitness_observer.rs`、`src/runtime/body_fitness_f4a_tests.rs`、`tests/body_fitness_birth.rs`、新しい `tests/body_fitness_birth_metabolism.rs` の6ファイル。mainのsrcには採用していない。

全suiteは **1311成功・0失敗・48 ignore**。`RUST_BACKTRACE=1 cargo test -- --nocapture` のstdout/stderrを `test_report.txt`、同じshellで捕捉した終了コードを `test_status.txt` に保存した。終了は2026-09-27 16:01:11 JST、exit 0。format、標準 `cargo clippy -- -D warnings`、`cargo check --all-targets` もexit 0。全検査後に410 sourceのhash不変を確認した。

[固定sceneの取得](body-fitness-birth-metabolism-results-20260927.md)では、出生のみONと出生・代謝ともONを各2回実行し、各mode内のWAVと対象reportが一致した。出生hopの子はBirthShared、翌hopはSourceRemovedとなり、既存Voiceの自己除去環境は維持した。frame 96の子energyは0.85603476と0.85717773、差0.00114297。出生位置とWAVはmode間同一。この限定接続を生存差、音色遺伝、実時間性能の証明とは扱わない。

## 固定した出所

| 固定物 | SHA-256 |
|---|---|
| 第十六版d source manifest | `01b8cc1c04aa9ffc2d79fc015f74852f7fb6d488215bff934a4394b8b4fddcaa` |
| 通常render binary | `309ab30716e83bab6e1c11efcd4506330df89774706d31f62e215549cd3575d8` |
| 第十五版b基準source manifest | `30d7d1b19289799a767bd02366d04556b5138779bcc5cedb1b37765d77cf9e43` |
| 初回登録 | `17c1cde406668c97785b7beec3590f3316a50bdd7e02e075a6a7fd54a0272fe2` |
| 固定環境のanchorを明記した取得前補足版 | `8f5092ebc70225b45087eb9a47cf51e415607b1bd43412c701bcbe94809d1ff9` |

worktree内の `target/integration-source-20260927-v16d/` にsource capsule、`target/birth-metabolism-baseline-v15b/` に基準manifestと第十五版b検証記録を保持する。`target/birth-metabolism-validation/` には全検査ログ・実行script・登録両版・固定render binary・一次取得対応表・成果物hash一覧を保存した。今回の取得は通常のdev buildによる決定的offline renderであり、release専有計時は行っていない。

全suiteの結合一次rawは `/home/shafi/lwrk/conchordal/target/body-fitness-birth-metabolism-build-20260927/runtime-birth-metabolism-evidence/registered-4162-1790492330438638435/`、配線前拒否rawは同じ親dirの `scenario-gates-4162-1790492330438609495/`。一次取得のsource manifestと245ファイルの個別hashを固定410ファイルに照合し、4回のrender binary hashも保存コピーと一致した。先行focused取得のsummaryも一次取得と一致した。

同じ全suiteで通常の単独出生・単独代謝・非Sine action回帰も実行した。それぞれ共通build先の `runtime-birth-evidence/normal-4104-1790492327743849133/`、`runtime-metabolism-evidence/run-4215-1790492344175668296/`、`runtime-action-ecology-evidence/run-4048-1790492281345884807/` にrawを保持する。

## 検査と修正履歴

新規検査はorigin/初回installの2件、observerの無音子・PCM欠落の2件、結合IR gateとbatch境界の2件、通常render結合と配線前拒否の2件。既存の自己除去の独立履歴比較、候補中心と実jitter後の直接積分も全suiteに含む。新しい各故障をすべて通常renderへ注入したものではなく、局所検査・実runtime取得・assertのみの境界を[結果](body-fitness-birth-metabolism-results-20260927.md)で区別した。

- 初回v16は旧試験補助関数へenumを渡した7箇所の型不一致で全target checkが失敗した。source manifest `833af342ebcb017126f809c356b7e8766decb7324e1299583d88f7a564c09b5b` と `initial-v16/` のログを保持する。
- v16bは新しい無音子試験で音響履歴の時計を進めず、出生sample 512に対して時計0となり失敗した。libは1215成功・1失敗・48 ignore。source manifest `35d46b9d4d344f0fe37fd4641d55c4120c5de8b6d9a0ae563d5cbc90d526655b` と `initial-v16b/` の全出力を保存。実runtimeと同じ順で無音1hopを処理する試験へ修正し、focused 1件成功を確認した。
- v16cは通常Rhaiが各Spawnと同eventに置くcrowding初期設定をgateが誤拒否した。lib 1216件と先行integrationは成功したが、結合取得は失敗。source manifest `ebd21f5d4e41611bb0a448888e6f804116808f803c1a8952722e68475b0a1c43`、`initial-v16c/` のログ・binary、失敗rawを保持した。同eventの直前Spawnと同populationに属する初期設定を一回だけ受理し、後続変更・別population・重複・Spawn前設定を拒否するよう修正した。
- 続くfocused取得では4renderが完了し再現したが、検査側が32byte配列のRecipe hashをu64と誤認した。`focused-schema-failure/` とrawを保持し、配列長と各byte範囲の検査に修正した。修正後のfocused 2件と最終全suiteを通過した。

以上の修正でseed・scene・係数・energy差の閾値は変更していない。レビューで指摘された既定OFFの毎hop heap確保はstack配列へ変更し、演奏終了後の空Populationはtail処理として区別した。複数Finishと、子の翌hopを確保しない短いFinishは配線前に拒否する。

既定OFF、48 kHz/512の決定的offline、初期一声と後続Consonance/Peak Field子一声に限定する。action/observation併用、instrument、親ありrespawn、動的解析は未受理。次の[親ありrespawn案](body-fitness-offline-respawn-draft-20260927.md)はまだ登録・実装前である。I11資源14/16、元のModal記憶参照0、非同期代謝、長期生態、音色遺伝、作者採用の未完は維持。第十五版bの専有計時を新版の実時間合格へ読み替えず、commit・pushも行っていない。
