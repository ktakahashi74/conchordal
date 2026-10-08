# 次のセッションの始め方

2026-10-08更新。現行計画の20単位は同日にすべて出口に達した。結果は[plan-current.md §7](plan-current.md#7-結果)に記録した。T-4は不成立、T-6は到来の重み1の採用と、検出の数値を選定できない問題の名指しによって閉じた。時間構造の再設計は始めていない。

次に決めるのは、未pushのmainをoriginへ送るかと、周期の確定・到来の予測の設計へ進むかである。設計案は単一段差Periodicの置換を含み、確定の条件と期待から次のeventの確率への対応が未定義である。20単位の終了を、その設計の採択や実装完了と読み替えない。

## 1. 始め方

統括はClaude Opusのセッションが行う。Emacsでこのcheckoutのagent-shell（Claude）を開き、§2の文書を読んで現状を確認する。新しい単位は、必要な作者判断の後に、作業範囲と出口を明記して割り当てる。2026-10-07の段0・段1を再実行する指示は使わない。

workerは `gpt-6.1-sol` の `xhigh`、KEIOアカウントのagent-shell bufferである。統括が `scripts/worker-shell.el` で起動、指示、状態確認を行う。以下の `T-1` は起動構文の例であり、済んだ単位の再実行指示ではない。

```sh
# 起動（返り値はbuffer名 "Codex worker T-1 @ conchordal"）
agent-emacsclient --eval '(progn (load "/home/shafi/lwrk/conchordal/scripts/worker-shell.el" nil t) (conchordal-worker-start "T-1"))'

# 指示。briefはfileに書き、送る文は短くする
agent-emacsclient --eval '(conchordal-worker-send "Codex worker T-1 @ conchordal" ".orchestration/units/T-1/brief.md を読み、その単位を実行せよ。")'

# 状態。各workerの (名前 アカウント 実行中か) を返す
agent-emacsclient --eval '(conchordal-worker-status)'
```

- モデルと努力度は、このEmacsのCodexの既定（`gpt-6.1-sol`、`reasoning_effort` = `xhigh`）から入る。アカウントは `scripts/worker-shell.el` がKEIOを指定する。指定しないと、このcheckoutのCodexはRIKENになる。
- 2026-10-07の試験起動で、KEIO（`CODEX_HOME=~/.codex-keio`）、`gpt-6.1-sol`、`xhigh` と、作者の表示中のwindowを変えないことを確認した。過去の試験・統括bufferが今も残っているとは扱わず、起動前に状態を確認する。
- workerの結果は、briefで指定したfileに書かせる。統括はそのfileを読む。完了の検知は `conchordal-worker-status` の「実行中か」と、結果fileの有無で行う。

## 2. 先に読むもの

1. [plan-current.md §7](plan-current.md#7-結果)：20単位の閉じ方と、計画の外へ残した課題。§1〜§6は計画時点の記録である。
2. 設計台帳[§9.3.58](../design-notes/technote-ledger.ja.md#realtime-acceptance-scope)：R-1の合格範囲、開始時の競合の修正、外乱下の不合格。[§9.3.59](../design-notes/technote-ledger.ja.md#temporal-rule-calibration-outcome)：音色・時間構造の選定結果と未決の設計。英語版にも同じ結果がある。
3. `.orchestration/HANDOFF.md`：統括の最新の引継ぎ。`.orchestration/status.md`は単位別の表、`.orchestration/records/decisions-20261007-orchestrator.md`は時刻付きの判断記録である。古い冒頭の更新日時や件数だけで状態を判断しない。
4. `AGENTS.md`のProgress and Validation Scope、Multi-agent Orchestration、Author Decision Requests、Git Operation Policyと、`.orchestration/units/COMMON.md`。

## 3. 計画終了時点の状態と、残るもの

2026-10-08の計画終了時点でmainは`20a6904`、未pushである。記録と再計算の仕組みは`work/t6-calibration-20261008-v1`（`ff9c856`）に残し、mainには入れていない。共有checkoutのHEAD・差分とbranchの状態は、着手時に改めて確認する。

新しいrendererはofflineのopt-inで、楽器では使えない。身体込みの出生・代謝と代表onset footprintも新rendererでは対応外である。既定化・楽器への接続、再出生への身体評価の拡張、揺らぎの語彙、H／RとT2の聴覚前処理の共有は、plan-current.md §7に残した判断・設計課題である。

## 4. 2026-10-07の開始手順の履歴

旧版は、未コミットの計画改訂をworktreeで固定してから、段0・段1のU-1〜U-4、T-1／T-2、B-1／B-2、M-1を割り当てる手順を示していた。その土台作業と後続の単位は、§2で参照した結果に置き換わった。旧版の開始時HEAD、bufferの残存状態、最初のwave案を、次の作業指示として使わない。2026-10-07の採択原文と根拠は`.orchestration/records/author-instruction-review-20261007-v1/`の`adopted.md`と`proposal.md`に残る。

## 5. 注意

- commitは共有checkoutで行わず、指定worktreeのbranchで行う。commit前のClippyとpre-commit hookを省略しない。hookは `cargo fmt --all` と `git add -u` を実行するため、追跡差分が意図した対象だけであることを先に確認する。`--no-verify` や `core.hooksPath` の上書きで迂回しない。
- gitに無い内容を持つworktreeは削除しない。2026-10-07にindexが全削除だった `f2-c2-*`、`f2-nominal-resource-prototype-20261007-v1`、`f2-shared-control-feature-prototype-20261007-v1` は、内容がディスク上にしか無かった。整理するときは現物を確認し、保全と作者の指示を優先する。
- 統括の記録は `.orchestration/` に置く（`status.md`、`units/<単位>/` のbriefと報告、`records/`、計時のlock）。`target/` には唯一の写しを置かない。2026-10-07 に共有checkoutで `cargo clean` が実行され、`target/` 以下の追跡外の記録が失われた。復元できたのは `adopted.md` と `proposal.md` だけである。
- workerの起動直後はsessionが未確立である。`agent-shell--state` の `(:session :id)` が入ってから `conchordal-worker-send` し、送信後は `~/.codex-keio/sessions/` のrolloutかbufferの伸びでturnの開始を確かめる。workerのbufferは `conchordal-worker-frame` で別frameに並べる。
- 計算前に `.orchestration/units/COMMON.md` の「メモリ」を読む。MemAvailableが12 GiB以上あることを確認し、所有jobだけへ8 GiBの上限を掛ける。user scopeが使える場合は `MemoryMax=8G`・`MemorySwapMax=0`、sandboxで使えない場合は `RLIMIT_AS` 等の代替を使う。端末やEmacsのscope、slice、`MemoryHigh` は変更しない。cargoの並列は8以下とし、PID・実際のlimit・exit・可能ならRSS peak・killの出所を記録する。
- `CARGO_TARGET_DIR` は系統ごとに一つを使い回す。単位ごとに新しいworktreeを作ってcacheを複製することはしない。
- 計算前は共有checkoutの `/home/shafi/lwrk/conchordal/scripts/timing_lock.sh check` で排他枠を確認する。worktreeの古いscriptは使わない。合否に使う計時は、1セルの試行の後に排他の時間枠で行う。
- R-1の実機測定は統括のshellから行った。workerのsandboxは音声deviceとuser busに届かず、`--play true` が初期化で止まりprocessが残った。取得・集計のscriptは `.orchestration/units/R-1/realtime-20261008/run_r1.py` と `analyze_r1.py` にある。
- 排他の時間枠は別projectのjobに効かない。R-1のBのv2では、別projectの検査jobによって全処理が約3倍遅くなり不合格だった。測定の前後にload averageと `journalctl --user` のjob開始・CPU消費を確認し、外乱ありの不合格と静かな状態での再測定を別々に保存する。
- `agent-emacsclient --eval` から確認dialogを出しうる `find-file-noselect` や `revert-buffer` を呼ばない。2026-10-08にGTK dialogがEmacs全体を止めた。呼出しには `timeout` を付ける。
- 作者へ判断を求めるときは、推奨ごとに、採ると発生する作業、採らないと消える作業、数値の由来、後から変える費用を書く。
- 作者の未決判断はworkerが決めない。pushと、gitに無い内容を失う削除は明示の指示を待つ。
