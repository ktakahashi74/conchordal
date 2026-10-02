# blanche の OOM 再発防止

## 撤回: 端末全体への制限は再利用しない

2026-10-01 21:35:12 と21:35:27 JST、systemd-oomd が2つの Ptyxis scope を丸ごと終了した。後者には共有 Emacs と複数のエージェント・テストプロセスが含まれていた。この文書で案内した端末全体の `MemoryHigh` 制限は撤回した。`run-guarded` の配置も解除した。以下の導入説明は事故時点の履歴であり、現行の運用手順ではない。

systemd-oomd のログは、ユーザー全体のメモリ圧力が82.03%と80.41%になり、設定された50%を20秒超えて継続したことを終了理由として記録している。端末に適用した4 GiB、および共有 Emacs の16 GiBという soft 上限が回収待ちと減速を発生させた。kernel の `oom_kill=0` でも systemd-oomd は scope 内の全プロセスを終了できる。`OOMPolicy=continue` もこの別経路の終了を防がない。

新設した全端末向け drop-in、既存端末の上限、共通 slice の上限を解除した。専用 F2 scope の8 GiB上限は変更せず、計測の継続を確認した。原因ログは `target/oom-prevention-20261001/oomd-incident.json`、解除記録は `terminal-limit-rollback.json`、元の検証記録は `WITHDRAWN_AFTER_OOMD_TERMINAL_KILLS` として保存した。

今後の制限対象は、所有が明確な計算ジョブだけとする。端末や Emacs 全体には適用しない。再設計では systemd-oomd の監視対象と memory pressure を検証に含める。事故前に終了した Python の用途は依然として特定できていない。

## 以下は撤回した導入の履歴

2026-10-01 20:38:30 JST、メモリ制限のない Ptyxis 端末の Python が約22.3 GiBの匿名メモリを消費し、ホスト全体の OOM で終了した。Python の処理内容は記録から特定できていない。同時刻の F2 Source5 計測は別の8 GiB・swap禁止の scope 内で継続していた。

## 起動方法

重い Python、Cargo、データ検証は次の起動方法を使う。

```bash
run-guarded python3 analysis.py
run-guarded cargo test -- --nocapture
```

`scripts/run_guarded.py` はジョブ全体を `resource-work.slice` の子 scope に置く。個別の `MemoryHigh=4G`、`MemoryMax=8G`、`MemorySwapMax=0` と、親 slice の合計16 GiB・swap禁止を実際の cgroup ファイルで確認してからコマンドを実行する。ホストの `MemAvailable` が12 GiB未満なら開始しない。Cargo のビルド並列数は2、Rust のテスト並列数は1に固定する。

PID、コマンド、作業ディレクトリ、実効上限、終了コード、メモリピーク、OOM 回数は `~/.local/state/resource-guard/<run-id>/run.json` に保存する。`XDG_STATE_HOME` を指定している場合はその下に保存する。開始拒否は終了コード75または78、コマンドの失敗は元の終了コードで返す。

`OOMPolicy=continue` により、scope 内の一プロセスの OOM を理由に端末や記録用プロセスまで systemd が終了させる動作を避ける。メモリ上限は引き続き有効で、カーネルが対象プロセスを終了させる。ラッパー自体が強制終了された場合は `scope_terminated` として失敗を保存し、取得できなかった終了コードやカウンターをゼロで埋めない。

## 端末からの直接起動

ユーザー設定の `ptyxis-spawn-.scope.d/50-memory-guard.conf` は、新規 Ptyxis scope にも4/8 GiB・swap禁止と同じ親 slice を適用する。直接 `python` を実行しても端末全体の上限が働く。ただし `systemd-run` などで別 scope に出る処理には、その scope 自身の制限が必要になる。

導入前から稼働している端末には、現在の匿名メモリを確認して個別上限を適用した。共有 Emacs と別案件の並列テストが動く端末1個だけは、実行中の作業を停止させないため `MemoryHigh=16G`、`MemoryMax=24G`、swap上限1 GiBの例外とした。ほかの既存端末は4/8 GiB・swap禁止。

既存 scope の所属 slice は移動しない。これらの端末は次の起動から共通の合計上限に入る。既存端末の所属と例外は `/run/user/1000/systemd/user/<scope>.d/90-existing-location.conf` に一時保存した。再起動後には残らない。`systemctl show` の表示だけでなく、`/sys/fs/cgroup` の実効上限を確認する。稼働中の登録済み F2 計測の上限や所属は変更しない。

## 制限に達したとき

失敗した取得や計測を自動再実行しない。記録から対象 PID・コマンドと OOM の有無を確認し、必要なメモリ量を調べる。一括読み込みや全件保持が原因なら逐次処理へ変更する。上限変更が必要な場合は、ホストの余裕と同時実行数を確認してから、そのジョブの設定と登録を改める。

設定ファイルの正本は `scripts/resource-guard/`。導入・動作確認の記録は `target/oom-prevention-20261001/` に保存する。
