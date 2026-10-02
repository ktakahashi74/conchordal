# 第十五版bの統合検証

隔離worktree `.worktrees/body-fitness-cold-parallel` に第十四版を引き継ぎ、冷cacheの候補準備を条件付きで2laneへ分割した。変更は `src/runtime/body_fitness_preparation.rs`、`body_fitness_action.rs`、新しい `body_fitness_parallel_tests.rs` の3ファイル。mainの通常実装には採用していない。

全suiteは **1303成功・0失敗・48 ignore**。`RUST_BACKTRACE=1 cargo test -- --nocapture` のstdout/stderrを `test_report.txt`、同じshellで捕捉した終了コードを `test_status.txt` に保存した。終了は2026-09-27 15:20:55 JST、exit 0。format、標準 `cargo clippy -- -D warnings`、`cargo check --all-targets`、release lib testのno-run構築もexit 0。全検査後に409 sourceのhash不変を確認した。

[専有取得](body-fitness-cold-parallel-results-20260927.md)は、非Sine身体/control変更のON/OFF、Sine 1/4 sourceのON/OFF、計6条件すべて合格。各条件10秒・938 hop、模擬出力先に限定した結果であり、実deviceや長時間の受入ではない。非Sineの新世代消費は登録期限frame 432までに成立した。第十三版b・第十四版の失敗取得は保持する。

## sourceと成果物

| 固定物 | SHA-256 |
|---|---|
| 第十五版b source manifest | `30d7d1b19289799a767bd02366d04556b5138779bcc5cedb1b37765d77cf9e43` |
| 両取得に使ったrelease paced test binary | `efb13e36c5ba24f8739c5c53a964447d1db696fb45c7c0e2f02ad753f0285224` |
| 第十四版基準source manifest | `3be6848c779cd915002af2c48e2ce87151037814f5bb3a2db57049e8eb6d48b8` |
| 実装前登録（両取得共通） | `0f21bed286fb0cfeae0bfbc43889926a7658e813a09de31874abfe7a91976632` |

worktree内の `target/integration-source-20260927-v15b/` に固定source、`target/cold-parallel-baseline-v14/` に基準、`target/cold-parallel-validation/` に全検査ログ・script・host前後確認・成果物hash一覧を保持する。固定実行ファイルとrawは `target/runtime-non-sine-v15b-20260927/`、`target/runtime-sine-four-v15b-20260927/`。両取得とも他のcargo/test/renderがない状態をホスト側で前後確認した。

全suiteの通常offline証拠は `/home/shafi/lwrk/conchordal/target/body-fitness-cold-parallel-build-20260927/` 以下に保持する。出生は `runtime-birth-evidence/normal-5574-1790489924506948216/`、代謝は `runtime-metabolism-evidence/run-5634-1790489927237340500/`、非Sine actionは `runtime-action-ecology-evidence/run-5518-1790489877870941123/`。これらは隔離版の通常配線の検証であり、main採用ではない。

## 局所検査と失敗履歴

[局所検証](body-fitness-cold-parallel-focused-results-20260927.md)は新規4件、既存準備4件、action 5件。全候補bit、cache順・統計、現環境score、実Voice判断とRNG、warm/重複/上限超過fallback、1/2/4 sourceの取消・回復を検査した。同一版の逐次参照との比較であり、旧版独立再実装との比較とは呼ばない。

新規試験の最初のcompile失敗（CloneのないVoice）と0件filter実行を成功件数から除外した。初回v15は同期型importに残った `#[cfg(test)]` により標準Clippyの通常lib compileが失敗した。初回source manifest `97c2cc3056460e44babdf857b7a38dec63b630e3f8989e40a1839861739de553` と `target/cold-parallel-validation/initial-v15/` を保持し、属性一行の修正後をv15bとして全検査した。失敗版の性能取得は行っていない。

取得後のSine要約scriptは、非Sine専用fieldの参照で一度KeyErrorとなった。要約側のfield一覧だけを修正した。固定source・raw・binaryは変更せず、性能試験も再取得していない。

今回の最大8計算threadは4 sourceの候補準備だけの上限で、解析・listener・出力等のthreadは別。cache 1 MiB/sourceもprocess全体のmemory上限ではない。定期reportに残らない身体変更後の冷job計算時間や正確な初回消費時刻は推定値で埋めない。

次は[通常offline出生と代謝の結合案](body-fitness-birth-metabolism-coupling-draft-20260927.md)の独立登録と実装。現版は併用を拒否する。F0〜F6全体、I11資源14/16、元のModal記憶参照0、実device、長期生態、作者採用の未完は維持する。commit・pushは行っていない。
