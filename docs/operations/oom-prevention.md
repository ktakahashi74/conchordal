# blanche の OOM 再発防止

## 撤回: 端末全体への制限は再利用しない

2026-10-01 21:35:12 と21:35:27 JST、systemd-oomd が2つの Ptyxis scope を丸ごと終了した。後者には共有 Emacs と複数のエージェント・テストプロセスが含まれていた。この文書で案内した端末全体の `MemoryHigh` 制限は撤回した。`run-guarded` の配置も解除した。以下では撤回した端末設定と、現行のジョブ用ラッパーを区別する。

systemd-oomd のログは、ユーザー全体のメモリ圧力が82.03%と80.41%になり、設定された50%を20秒超えて継続したことを終了理由として記録している。端末に適用した4 GiB、および共有 Emacs の16 GiBという soft 上限が回収待ちと減速を発生させた。kernel の `oom_kill=0` でも systemd-oomd は scope 内の全プロセスを終了できる。`OOMPolicy=continue` もこの別経路の終了を防がない。

新設した全端末向け drop-in、既存端末の上限、共通 slice の上限を解除した。専用 F2 scope の8 GiB上限は変更せず、計測の継続を確認した。当時の原因ログと解除記録は `target/oom-prevention-20261001/` に保存されていたが、この directory は現在存在しない。元の検証は撤回済みであり、その導入を再利用しない。

今後の制限対象は、所有が明確な計算ジョブだけとする。端末や Emacs 全体には適用しない。再設計では systemd-oomd の監視対象と memory pressure を検証に含める。事故前に終了した Python の用途は依然として特定できていない。

2026-10-13：`/tmp`はtmpfsであり、検査の出力を残すとRAMとswapを消費する。Cargoのtest・buildと描画jobは、disk上の`TMPDIR=/home/shafi/.cache/conchordal-test-tmp`へ一時出力を置く。通った検査は、そのjobが作った自分の出力を消す。失敗時は調査に必要な証拠だけを既存の有界な保持規則で残し、使用中や別jobの出力、手動の取得物は消さない。

## 事故の経緯

2026-10-01 20:38:30 JST、メモリ制限のない Ptyxis 端末の Python が約22.3 GiBの匿名メモリを消費し、ホスト全体の OOM で終了した。Python の処理内容は記録から特定できていない。同時刻の F2 Source5 計測は別の8 GiB・swap禁止の scope 内で継続していた。

## 現行のジョブ用ラッパー

共有端末ではなく、明示的に所有する計算ジョブを `scripts/run_guarded.py` で起動する。撤回した `run-guarded` コマンドの配置を前提にしない。

```bash
python3 scripts/run_guarded.py -- python3 analysis.py
python3 scripts/run_guarded.py -- cargo test -- --nocapture
```

`scripts/run_guarded.py` の2026-10-13のジョブ用配置は、全 worker の合計上限40 GiBの `agent-workers.slice` と、その子で合計16 GiB・swap禁止の `agent-workers-guarded.slice` である。ジョブ全体は後者の子 scope に置く。個別の `MemoryHigh=4G`、`MemoryMax=8G`、`MemorySwapMax=0` と、直上 slice の合計16 GiB・swap禁止を実際の cgroup ファイルで確認してからコマンドを実行する。ホストの `MemAvailable` が12 GiB未満なら開始しない。Cargo のビルド並列数は2、Rust のテスト並列数は1に固定する。

PID、コマンド、作業ディレクトリ、実効上限、終了コード、メモリピーク、OOM 回数は `~/.local/state/resource-guard/<run-id>/run.json` に保存する。`XDG_STATE_HOME` を指定している場合はその下に保存する。開始拒否は終了コード75または78、コマンドの失敗は元の終了コードで返す。

`OOMPolicy=continue` により、scope 内の一プロセスの OOM を理由に端末や記録用プロセスまで systemd が終了させる動作を避ける。メモリ上限は引き続き有効で、カーネルが対象プロセスを終了させる。ラッパー自体が強制終了された場合は `scope_terminated` として失敗を保存し、取得できなかった終了コードやカウンターをゼロで埋めない。

## 端末からの直接起動

以下は撤回した設定の履歴であり、現行の起動手順ではない。端末全体への制限は再導入しない。追跡されていた `ptyxis-memory.conf` template も削除した。

当時はユーザー設定の `ptyxis-spawn-.scope.d/50-memory-guard.conf` が新規 Ptyxis scope にも4/8 GiB・swap禁止と同じ親 slice を適用していた。直接起動した Python も端末全体の制限に入り、これが共有セッションを巻き込んだ。

導入前から稼働している端末には、現在の匿名メモリを確認して個別上限を適用した。共有 Emacs と別案件の並列テストが動く端末1個だけは、実行中の作業を停止させないため `MemoryHigh=16G`、`MemoryMax=24G`、swap上限1 GiBの例外とした。ほかの既存端末は4/8 GiB・swap禁止。

既存 scope の所属 slice は移動しない。これらの端末は次の起動から共通の合計上限に入る。既存端末の所属と例外は `/run/user/1000/systemd/user/<scope>.d/90-existing-location.conf` に一時保存した。再起動後には残らない。`systemctl show` の表示だけでなく、`/sys/fs/cgroup` の実効上限を確認する。稼働中の登録済み F2 計測の上限や所属は変更しない。

## 制限に達したとき

失敗した取得や計測を自動再実行しない。記録から対象 PID・コマンドと OOM の有無を確認し、必要なメモリ量を調べる。一括読み込みや全件保持が原因なら逐次処理へ変更する。上限変更が必要な場合は、ホストの余裕と同時実行数を確認してから、そのジョブの設定と登録を改める。

現行ラッパーの設定と実効上限の検証は `scripts/run_guarded.py` が持つ。`scripts/resource-guard/resource-work.slice` は旧構成の参照であり、現行の `agent-workers-guarded.slice` の正本ではない。ジョブの実行記録は上記の state directory に残し、消去される `target/` を唯一の保存先にしない。
