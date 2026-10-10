# 次のセッションの始め方

2026-10-10更新。現行計画の20単位は2026-10-08にすべて出口に達した。結果とその後の追加作業は[plan-current.md §7](plan-current.md#7-結果)に記録した。T-4は不成立、T-6は到来の重み1の採用と、検出の数値を選定できない問題の名指しによって閉じた。N-7の夜の再測定で昼のB′不合格を解消し、その後、T-7の最終sourceによるA・B・A′・B′の四構成の実機再測定も合格した。過去の不合格は履歴として保持している。

作者は2026-10-10に動的代謝の受入を改訂してN-7を受け入れた。N-7は同日にmainへ統合され、originへpush済みである。時間構造は、原典の図の数値再現を一律の門にした誤りを撤回し、T-6kで整数比の階層の設計を閉じ、T-7で本体に実装した。T-7も同日にmainへ統合し、最終検査の後、11:50にoriginへpushした。作者の試聴では周期が合い同期しているが、位相が揃って縮退し、音楽的ではなかった。残るのは、拍に乗るVoiceの位相がVoiceどうしの関係から分かれる仕組みの設計問題である。20単位の終了、実装・統合・push、作者の音楽的な評価を区別する。

作者は同日、慣れを既定で有効にする判断も採った。この変更は[H-3のbranch](#habituation-default-branch-20261010)で用意し、mainへの統合とpushは指示待ちである。慣れ有効の四構成の実時間は既に合格しており、既存の科学的な対照では無効を明示する。

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

1. [plan-current.md §7](plan-current.md#7-結果)：20単位の閉じ方、[追加作業](plan-current.md#additional-work-20261009)のN-6・N-7、T-7の実時間合格と統合・push、残るもの。§1〜§6と20単位の表は計画時点・終了時点の記録である。
2. 設計台帳[§9.3.58のT-7の実機再測定](../design-notes/technote-ledger.ja.md#t7-meter-remeasurement-20261010)、[統合・push完了の追記](../design-notes/technote-ledger.ja.md#integration-push-completed-20261010)、[動的代謝の受入改訂](../design-notes/technote-ledger.ja.md#body-metabolism-acceptance-revision-20261010)：四構成の最新の合格、最終検査、N-7の波形の一致範囲、二つの動的分岐、合否と診断の分離。[§9.3.59の文化層の訂正](../design-notes/technote-ledger.ja.md#meter-tempo-prior-20261010)と[作者の試聴・位相縮退](../design-notes/technote-ledger.ja.md#metric-phase-degeneracy-audition-20261010)：T-6kの設計、T-7の実装、既存の作曲者の語彙への接続と、残る設計問題。英語版にも同じ結果がある。
3. `.orchestration/HANDOFF.md`：統括の最新の引継ぎ。`.orchestration/status.md`は単位別の表、`.orchestration/records/decisions-20261007-orchestrator.md`は時刻付きの判断記録である。古い冒頭の更新日時や件数だけで状態を判断しない。
4. `AGENTS.md`のProgress and Validation Scope、Multi-agent Orchestration、Author Decision Requests、Git Operation Policyと、`.orchestration/units/COMMON.md`。

## 3. 現在の状態と、残るもの

2026-10-09にmainを二度originへpushし、2026-10-10も10:52と11:50にpushした。N-7とT-7はmainへ統合済みである。T-7の最終sourceの実機計測は統合前に済み、A・B・A′・B′とも宣言範囲で合格した。統合後、mainと同じ最終sourceで統括がfmt・Clippy・全テストを実行し、1,027 passed、0 failed、34 ignored、exit 0を確認した（2026-10-10 11:50:29）。mainのHEAD・差分、originとの関係、branchの状態は着手時に確認する。校正の記録と再計算の仕組みは`work/t6-calibration-20261008-v1`（`ff9c856`）に残し、mainには入れていない。これはT-7の本体実装とは別の校正用の記録である。

新しいrendererは楽器でも`render_prototype = true`で選べる。Sine／Harmonicで身体込みの出生8 family、代謝、代表onset footprintが使えるが、Modalは対応外である。旧rendererの再出生の最終選択は方式とfamilyの対応する組だけ身体込みで評価し、新しいrendererではnativeの参照分布がないため身体評価を使わない。

N-7のsourceによる夜の測定は、A′・B′とも宣言範囲で合格し、昼のB′不合格を解消した。A・BはN-7では旧rendererが変わらずWAVが一致するため測り直していなかったが、T-7の最終sourceでは四構成すべてを再測定した。最新の演奏別p99最大はAが3,912 µs（09）、Bが6,769 µs（10）、A′が7,497 µs（09）、B′が9,258 µs（09）で、全48演奏が終了0、出力不足とcallback errorは0だった。BとB′の個々のhopには予算超過があるが、採択したp99と出力不足の基準は満たした。別の日・別の負荷との値の差をmeterの変更の費用とは読まない。取得範囲と数値は台帳[§9.3.58のT-7再測定](../design-notes/technote-ledger.ja.md#t7-meter-remeasurement-20261010)にある。既定は旧rendererのままで、新しいrendererを既定にするか、Modalの対応、nativeの再出生の最終選択の参照が残る。単音holdのPCM一致を波形全体の旧renderer互換へ広げない。

動的代謝の合否は、静的照合（既存上限0.025）、更新の順番と漏れのなさ、同じseedでの再現性で判定する。動的比較は件数と最初の分岐の原因の報告として残す。更新数2の設定は変更していないが、seed 42での全件一致で選んだ根拠は弱くなり、見直しが残る。静的上限を動的scoreの誤差上限にはしない。別seed・別étudeでの発生率や生態の偏りは未検証である。

時間の期待は拍と間隔の二択にせず、既存の作曲者の語彙でどれだけ拍に乗るかを選ぶ。T-6kで整数比の階層の設計を閉じ、T-7で本体への実装と検査を済ませた。初回の確認前は三層の確からしさと比を0に保つ。2026-10-10の作者の指摘で、tempoの好みを文化層の事前分布と整理し、重みを既存の`meter_stability`（既定0）から取る形に直した。内部定数の試聴選択は残っていない。最終sourceの実機計測の入力・取得・集計は`.orchestration/units/T-7/realtime/`に保存し、四構成の合格は台帳[§9.3.58](../design-notes/technote-ledger.ja.md#t7-meter-remeasurement-20261010)に記録した。次の一回型の到来項は既定で無効、検出の数値は未採用のままである。作者の試聴では周期と同期は成立したが、位相が揃って縮退した。拍位相＋作曲者の`microtiming`という合わせ先を持ち、Voiceどうしの関係から位相が分かれる仕組みがないことが残る設計問題である。コードの事実と統括の未検証の見立ては、台帳[§9.3.59の試聴記録](../design-notes/technote-ledger.ja.md#metric-phase-degeneracy-audition-20261010)で区別する。解き方はこの記録では定めていない。

揺らぎの語彙も未決である。H／RとT2の聴覚前処理の共有は計画しており、作者の2026-10-09の指示は実施時期の延期であって取り下げではない。これらは[plan-current.md §7](plan-current.md#7-結果)で追跡する。

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

<a id="habituation-default-branch-20261010"></a>

### 2026-10-10：慣れの既定有効化をbranchで用意

作者は慣れ（`[psychoacoustics.habituation]`）を既定で有効にする判断を採った。
`main`の`1f55a45`を基準に、`work/h3-habituation-default-20261010-v1`へ実装と検査
（`6a7b6b2`）、文書同期を別commitで用意した。mainへの統合とpushは行っていない。
次の統合は作者の指示を待つ。既存の5秒・8秒・0.25と慣れの式、上位の12 étude、Rhaiの語彙は変えていない。
科学的な対照とbit一致の検査には明示offを使い、凍結済みの登録値・hash・結果を保持した。

慣れを有効にした四構成は、統括の実機取得で既に合格している。演奏別p99の最大は
Aが5,896 µs、Bが6,196 µs、A′が7,442 µs、B′が9,308 µsで、採る全48演奏が終了0、
出力不足とcallback errorは0だった。初回B′の不合格、B・A′のメモリ入口での中断、
取り直しの窓とA′の外乱を含む経過は、台帳[§9.3.60](../design-notes/technote-ledger.ja.md#habituation-default-on-20261010)に記録した。
実時間は取得し直していない。branchでの全テストと既定設定による12 étudeの全長offline描画の記録は
`.orchestration/units/H-3/report.md`にある。20単位の分母を増やさず、作者の試聴評価と実時間の合格を分ける。
