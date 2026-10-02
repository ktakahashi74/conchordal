# 第十四版の統合検証

隔離worktree `.worktrees/body-fitness-recovery` に、旧世代準備の取消、代表スペクトルのscratch再利用、通常offline初回Field出生を統合した。mainの通常実装には採用していない。

全suiteは **1299成功・0失敗・48 ignore**。`RUST_BACKTRACE=1 cargo test -- --nocapture` のstdout/stderrを `test_report.txt`、同じshellで捕捉した終了コードを `test_status.txt` に保存した。終了は2026-09-27 14:55:24 JST、exit 0。format、標準 `cargo clippy -- -D warnings`、`cargo check --all-targets`、release lib testのno-run構築もexit 0。全検査後に固定408 sourceのhash不変を再確認した。

専有の非Sine実時間試験は **OFF合格・ON不合格、exit 101**。全suite成功とは別判定。取消は1 hopで応答したが、新身体135候補の冷計算に約1.91秒かかり、消費frame 480で期限432を超えた。[取得結果](body-fitness-recovery-results-20260927.md)にrawと判断を記録した。

## sourceと成果物

| 固定物 | SHA-256 |
|---|---|
| 第十四版source manifest | `3be6848c779cd915002af2c48e2ce87151037814f5bb3a2db57049e8eb6d48b8` |
| release paced test binary | `360c292735d8fab7041548c24c6b3ab131706fe3f477b8c7f1c52529e6a5a78f` |
| 第十三版b基準source manifest | `2bc58ef9bffdb12eb50c042a1bcc678d9197d2d081f9d27d60b67bd263823b61` |

第十三版bの基準hashの正本は `target/recovery-baseline-v13b/source-manifest.json` とその検証記録。source capsuleは `target/integration-source-20260927-v14/`。全検査ログ、検査script、host前後確認、成果物hash一覧は `target/recovery-validation/`。実時間の実行ファイルコピーと取得は `target/runtime-non-sine-v14-20260927/`。いずれも隔離worktree内の相対path。

通常offlineの出生・代謝・actionのrawは、共通build先 `/home/shafi/lwrk/conchordal/target/body-fitness-recovery-build-20260927/` の各 `*-evidence/` に保持。最終全suiteの出生rawは `runtime-birth-evidence/normal-4987-1790488396584343744/`、代謝rawは `runtime-metabolism-evidence/run-5047-1790488399265309932/`、非Sine action rawは `runtime-action-ecology-evidence/run-4941-1790488344087545051/`。

## 検証範囲と失敗履歴

- [取消の局所検査](body-fitness-cancellation-results-20260927.md)は5件成功。最初の固定poll回数による試験失敗は対話のtool出力のみで、生ログファイルは未保存。最終全suiteには修正版を含む。
- [スペクトル比較](body-fitness-spectral-reuse-results-20260927.md)は封印旧版の隔離コピーと48 kHz/512の12条件×72 frameでPCM・scan・massがbyte一致。旧コピー405ファイルを事後照合し、差分はdump試験を追加したbody_footprint.rsだけ。取得時のbinaryそのものは未保存で、source監査・raw・build/testログを保持する。
- [通常offline出生](body-fitness-offline-birth-results-20260927.md)は既定falseの独立flag。初回spacing 1.0 ERB条件の判定誤りと、同時flag拒否の検査順序不備による失敗を保持。v2 spacing 0のOFF/ON各二回は決定一致。別Toneと直接積分の数値oracleを候補中心と実jitter後で検査した。このsceneでは旧評価と同じ出生位置・音声であり、出生位置の差は示さない。

通常出生は一つの初期環境Voiceと、後のField子一声に限定。Gap/Uniformは理由を報告して旧経路を使う。代謝/action/観測flagとの同時ON、途中の解析更新、同一Populationへの再spawn、respawn等は今回の範囲外として拒否する。epoch 0は静的な単一解析構成の識別子であり、一般的な再構成追跡の実証ではない。

F4/F5/F6全体の完了、実deviceの資源受入、長期生態、作者採用へ読み替えない。I11資源14/16、元のModal記憶参照0等の既存未達は維持。commit・pushは行っていない。
