# 次のセッションの始め方

2026-10-07 作成。計画の見直しを行ったセッションからの引継ぎである。計画の正本は [plan-current.md](plan-current.md)。

## 1. 始め方

統括はClaude Opusのセッションが行う。Emacsでこのcheckoutのagent-shell（Claude）を新しく開き、次の文を渡す。

```text
docs/roadmap/next-session.md と docs/roadmap/plan-current.md を読み、統括として段0と段1を進めて。
workerは scripts/worker-shell.el で立ち上げ、互いに独立な単位は並行で進める。
```

workerは `gpt-6.1-sol` の `xhigh`、KEIOアカウントのagent-shell bufferである。統括が次のように起動し、指示し、状態を見る。

```sh
# 起動（返り値はbuffer名 "Codex worker T-1 @ conchordal"）
agent-emacsclient --eval '(progn (load "/home/shafi/lwrk/conchordal/scripts/worker-shell.el" nil t) (conchordal-worker-start "T-1"))'

# 指示。briefはfileに書き、送る文は短くする
agent-emacsclient --eval '(conchordal-worker-send "Codex worker T-1 @ conchordal" ".orchestration/units/T-1/brief.md を読み、その単位を実行せよ。")'

# 状態。各workerの (名前 アカウント 実行中か) を返す
agent-emacsclient --eval '(conchordal-worker-status)'
```

- モデルと努力度は、このEmacsのCodexの既定（`gpt-6.1-sol`、`reasoning_effort` = `xhigh`）から入る。アカウントは `scripts/worker-shell.el` がKEIOを指定する。指定しないと、このcheckoutのCodexはRIKENになる。
- 2026-10-07 に1本を試験起動し、KEIO（`CODEX_HOME=~/.codex-keio`）、`gpt-6.1-sol`、`xhigh` で立ち上がること、作者の表示中のwindowを変えないことを確かめた。その試験用buffer `Codex worker test @ conchordal` が残っている。使うか閉じるかは任意である。
- 旧統括のbuffer（`Codex Agent @ conchordal` と `Codex Agent @ conchordal<2>`）は作者が止めてある。指示を送らない。
- workerの結果は、briefで指定したfileに書かせる。統括はそのfileを読む。完了の検知は `conchordal-worker-status` の「実行中か」と、結果fileの有無で行う。

## 2. 先に読むもの

1. [plan-current.md](plan-current.md)：原則、範囲、単位と順序、未決の作者判断。
2. `.orchestration/records/author-instruction-review-20261007-v1/adopted.md`：2026-10-07 の作者の返答の原文と採択範囲。
3. `.orchestration/records/author-instruction-review-20261007-v1/proposal.md`：変更案 C-01〜C-19 と D-01〜D-14 の根拠。
4. `AGENTS.md` の Progress and Validation Scope、Multi-agent Orchestration、Author Decision Requests。

## 3. 現在の状態

- `main` = `origin/main` = `d873986`。未pushのcommitはない。
- 未コミットの変更がある。本日の反映分は、`AGENTS.md`（評価の規則、判断依頼の提示、体制）、設計台帳の日英（機構選択規則の条件3）、旧計画5文書の冒頭の案内行、新規の `docs/roadmap/plan-current.md`、本書、`scripts/worker-shell.el`。旧統括が残した未コミットの変更（`AGENTS.md` の Progress and Validation Scope 節、音色計画、身体module契約、作者判断記録）も同じ作業木にある。
- 済んだ単位：U-5（容量。残り約300 GB）。
- 許可済み：U-3（小修正4件）のcommitとmainへの統合。pushは別に確認する。
- 作者の判断が要るもの：plan-current.md §4 の表。U-1（保全のcommit）もここに入る。
- 2026-10-07 の追加採択（plan-current.md に反映済み）：temporal本体を消費者のある経路に絞る、bodyの移動と代謝は最終候補と現在位置だけ身体採点、tailは振幅の下限で退役。

## 4. 最初に行うこと：本日の変更のcommit

作者が2026-10-07に許可した（「次のセッションの最初に、worktree branch経由でまとめてcommitする」）。workerを並行で動かす前に、基準をgitに固定する。pushは別に確認する。

- 対象は§3の未コミットの変更すべてである。旧統括が残した変更（`AGENTS.md` の Progress and Validation Scope 節、音色計画、身体module契約、作者判断記録）と本日の変更は同じfileに混ざっているので、一つのcommitにまとめ、その旨をcommit messageに書く。
- 共有checkoutでは commit しない。mainから新しいworktree branchを作り、共有checkoutの差分（`git diff` と未追跡の新規file 3件）をそこへ移してcommitする。commitの前に `cargo clippy -- -D warnings` を通す。
- その後、共有checkoutのmainをそのcommitへ進める。共有checkoutには同じ内容が未コミットで残っているので、進める前に作業木の差分がcommitと一致することを確かめてから、作業木を戻す。stashを使う場合は、他のセッションと共有である点に注意する。
- `.claude/` と `.codex` は未追跡のままでよい。

## 5. 最初のwaveの割当案

cargoを使わない単位は、すべて並行できる。

| 単位 | 作業 | 結果を書くfile |
|---|---|---|
| U-2 | 台帳に二層の規則と原則の節を起こす。参照されているanchorを実在させる。 | 台帳への差分 |
| U-4 | I12bの27行の結果をmilestonesに記録し、ownerの寿命の課題を起票する。 | milestonesへの差分 |
| T-1 | temporalの経路に要るmoduleと、研究branchへ移すmoduleの一覧。 | `.orchestration/units/T-1/report.md` |
| M-1 | Phase 3 の前提（励振への写像、開区間の法則、temporal側消費者の扱い、退役下限）。 | `.orchestration/units/M-1/report.md` |
| B-2 | 保存済みの全候補表に対する代理式の値の計算と、選択分布のずれ。 | `.orchestration/units/B-2/report.md` |

cargoを使う単位は、同時に2本までとする。

| 単位 | 作業 | 備考 |
|---|---|---|
| U-3 | 小修正4件を一件ずつmainへ入れる。 | 統合は直列。件ごとに全テストとClippy。 |
| B-1 | 自己除去の近道を参照器と照合する。 | 隔離worktreeの参照器とfixtureを使う。 |
| T-2 | I11-1の有効化の入口を一つにする。 | 無効時の出力のbit一致を確かめる。 |

U-1（要るworktree 4本の保全）は、commitの許可を作者に確認してから行う。

## 6. 注意

- commitは共有checkoutで行わない。worktreeのbranchで行う。pre-commit hookがfmtと `git add -u` を走らせる。
- indexが全削除の状態のworktreeが5本ある（`f2-c2-*` の3本、`f2-nominal-resource-prototype-20261007-v1`、`f2-shared-control-feature-prototype-20261007-v1`）。内容はディスク上のfileにしかないので、gitの操作や削除をしない。
- 統括の記録は `.orchestration/` に置く（`status.md`、`units/<単位>/` のbriefと報告、`records/`、計時のlock）。`target/` には唯一の写しを置かない。2026-10-07 に共有checkoutで `cargo clean` が実行され、`target/` 以下の追跡外の記録が失われた。復元できたのは `adopted.md` と `proposal.md` だけである。
- workerの起動直後はsessionが未確立である。`agent-shell--state` の `(:session :id)` が入ってから `conchordal-worker-send` し、送信後は `~/.codex-keio/sessions/` のrolloutかbufferの伸びでturnの開始を確かめる。workerのbufferは `conchordal-worker-frame` で別frameに並べる。
- 計算を走らせるworkerは、所有するjobだけにメモリの上限を掛ける（`.orchestration/units/COMMON.md` の「メモリ」）。端末やEmacsのscopeには掛けない。
- `CARGO_TARGET_DIR` は系統ごとに一つを使い回す。単位ごとに新しいworktreeを作ってcacheを複製することはしない。
- 合否に使う計時は `scripts/timing_lock.sh` の排他の時間枠で行い、その前に1セルを試行する。
- 作者へ判断を求めるときは、推奨ごとに、採ると発生する作業、採らないと消える作業、数値の由来、後から変える費用を書く。
