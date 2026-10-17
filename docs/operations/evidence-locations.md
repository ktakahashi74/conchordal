# 証拠の所在と保存方針

新しい記録の根拠は、gitで追跡する場所か、内容のハッシュを併記した保管場所を指す。
小さい要約、採択した判断、検査の結果と限界は追跡する文書に残す。
大きい音声、raw出力、source capsuleなどは外部に保存し、取り出せる保管場所とSHA-256を要約に記す。
ハッシュは同一性の確認に使うものであり、それだけで内容を復元できるわけではない。

過去の参照は当時の記録として残す。見つからない記録を再取得済み・合格済みと読まず、
保存した要約から分かることと、元の出力がなければ確認できないことを区別する。

## 既知の消失と参照の制約

- **`target/`**。2026-10-07の`cargo clean`で当時のディレクトリ全体が失われた。
  過去の文書が指す取得物、旧orchestration board、判断の記録なども含む。
  同じ名前のディレクトリが後で作られていても、以前の記録の復元を意味しない。
- **`.orchestration/`**。追跡外のmachine-localな作業記録であり、別のmachineへcloneしても得られない。
  単位のbriefやreport、試聴用の材料への参照は、そのmachineで所在を確認する必要がある。
  そこだけにある事実を現行計画の根拠に使う場合は、要点を追跡する文書へ写す。

計画は[現行計画](../roadmap/plan-current.md)、設計の記録は
[台帳](../design-notes/technote-ledger.ja.md)と[設計ノートの索引](../design-notes/README.md)から辿る。
この方針は過去の科学的判定を変更せず、消失した出力の回復も主張しない。

## 参照が多い文書

2026-10-17、整理前の基準commit `090f78c`で測定した。
以下は追跡されたMarkdownに対する`git grep -c`の上位10本であり、件数は一致した行数である。
同じ行にある複数のpathは1行と数え、コードとして引用したpathも含む。
元データの存在を検査した件数ではない。今回追加した索引や、移動後の履歴による重複は含めない。

### `target/`への参照

測定コマンドは`git grep -c 'target/' -- '*.md'`。128本、合計1,235行。

- [docs/roadmap/temporal-dcc/i10-body-outcome.md](../roadmap/temporal-dcc/i10-body-outcome.md)。190行。
- [docs/design-notes/parallel-continuation-20260927.md](../design-notes/parallel-continuation-20260927.md)。122行。
- [docs/roadmap/temporal-dcc/milestones.md](../roadmap/temporal-dcc/milestones.md)。75行。
- [docs/roadmap/phrase-expectation-evidence-2026-09-06.md](../roadmap/phrase-expectation-evidence-2026-09-06.md)。70行。
- [docs/design-notes/technote-ledger.md](../design-notes/technote-ledger.md)。67行。
- [docs/roadmap/temporal-dcc/a1-i11-audition.md](../roadmap/temporal-dcc/a1-i11-audition.md)。62行。
- [docs/design-notes/technote-ledger.ja.md](../design-notes/technote-ledger.ja.md)。58行。
- [docs/roadmap/temporal-dcc/m0.md](../roadmap/temporal-dcc/m0.md)。57行。
- [docs/roadmap/temporal-dcc/i11-onset-comparison.md](../roadmap/temporal-dcc/i11-onset-comparison.md)。49行。
- [docs/roadmap/manifesto-alignment-and-beta.md](../roadmap/manifesto-alignment-and-beta.md)。39行。

### `.orchestration/`への参照

測定コマンドは`git grep -c '\.orchestration/' -- '*.md'`。6本、合計128行。

- [docs/design-notes/technote-ledger.md](../design-notes/technote-ledger.md)。50行。
- [docs/design-notes/technote-ledger.ja.md](../design-notes/technote-ledger.ja.md)。50行。
- [docs/roadmap/next-session.md](../roadmap/next-session.md)。14行。
- [docs/roadmap/plan-current.md](../roadmap/plan-current.md)。10行。
- [docs/roadmap/temporal-dcc/milestones.md](../roadmap/temporal-dcc/milestones.md)。2行。
- [AGENTS.md](../../AGENTS.md)。2行。

