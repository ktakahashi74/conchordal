# Conchordal 設計・コード引き算 ＆ マニフェスト1周年改訂 実行指示書

作成日：2026-11-21  
完了目標：**2026-12-25**（マニフェスト初公開 2025-12-25 から1周年）  
参照正本：[現行アーキテクチャ正本](../architecture-current.md) ／ [棚卸し台帳](simplification-inventory-2026-11-21.md)

---

## 0. 本指示書の目的と原則

Conchordal は `origin/main`（`dee5a91d`）時点で `src/` **181,191 行**・Rhai **186 エントリ**・worktree **206 個（143 GB）** に成長した。本指示書は、**2026-12-25 のマニフェスト1周年改訂**に向けて、マニフェストに関わらない設計・コードの「引き算と構造整理」を先行完了させ、並行してマニフェストを磨くための実行手順を定める。

### 絶対に守る 4 原則
1. **引き算期間に新原理・新機構を足さない**：進行中の響き Phase 2（`body-phase2-20261121`）の試聴決着を唯一の例外とし、マニフェストに関わる機能追加は 12/25 以降に回す。
2. **「設計判断（削除・隔離）」と「純粋リファクタリング（構造整理）」を混ぜない**：別 commit・別単位で実行する。
3. **純粋リファクタリングは native PCM bit 一致を絶対ゲートにする**：音が変わった場合は即停止して原因を特定する。
4. **作者判断の領域を越えない**：機能の廃止（分類 C）やマニフェストの文意変更は必ず作者承認を経てから実行する。

---

## 1. 主線：設計・コード改善の実行単位（Unit S0〜S4）

### Unit S0：足場固め・正本同期・worktree 整理（目安：11/21〜11/25）

**2026-11-21 の記録**。足場の文書と棚卸しは `46b172c7d04663da7f2fdb8c5fe3cfd910d79411`。共有 checkout のローカル `main` は `04c6ba7e`、登録 worktree は197本。

**2026-11-22 の記録（SIMP-S0）**。
- 共有 checkout の GUI 3ファイルの差分は、追加行がすべて `origin/main` にあることを確かめて捨て、ローカル `main` を `351bac99` へ fast-forward した。`scripts/worker-shell.el` の差分はこの単位の commit に入れた。
- 登録 worktree 197本を削除した（作業中の可能性がある `simplify-20261121` は残した）。
  - 未 commit の変更があった137本は、乗っていた branch の先端（69本）か `archive/wt-<名前>`（68本）へ未検証の退避 commit にした。どの branch からも辿れない detached HEAD の1本にも `archive/wt-<名前>` を付けた。
  - worktree 内の取得記録（`target/` の cargo 以外の子、worktree 内の `.orchestration/`、`test_report.txt`／`test_status.txt`、5 MB を超える未追跡ファイルと音声）と、登録の外れた7つのディレクトリは、`.orchestration/records/worktree-leftovers-20261122/` へ移した（約 82 GB、同じファイルシステム内の移動）。消したのは cargo のビルド出力と生成物だけである。
  - 処置の一覧、使ったスクリプト、前後の ref は `.orchestration/units/SIMP-S0/` にある。
- 残る worktree は共有 checkout と `simplify-20261121` の2本。`simplify-20261121` は 11-22 の全曲照合（S4 の項）に使った後に外し、共有 checkout の1本になった。`.worktrees/alife-fitness/conc-paper-2026` は論文 repo の worktree なので触れていない。

**前提**：`origin/main` は `dee5a91d`、ローカル `main` は `04c6ba7e`（未 commit 4ファイルあり）。

**手順**：
1. **ローカル `main` の同期**
   - `scripts/worker-shell.el` の差分（`gpt-6.2-sol` 固定）だけを patch 退避する。
   - `src/ui/plots.rs`・`src/viewdata_web.rs`（`origin/main` と一致）および `src/ui/windows.rs`（`origin/main` が上位互換）を整理し、ローカル `main` を `origin/main`（`dee5a91d`）へ fast-forward する。
   - `scripts/worker-shell.el` の変更が必要なら再適用して commit する。
   - `docs/roadmap/next-session.md` の古い「main 未統合」記述を最新化する。
2. **進行中 worktree の決着**
   - `body-phase2-20261121`（DR-39 `motion` / DR-41 `resonance`, `attack_noise`）：作者試聴を完了し、採用分のみ `main` へ commit/push。不採用分と `room-stage1-20261119` の DR-40 は研究ブランチへ退避 commit して未 commit ゼロにする。
   - `dr1-consolidate-20261119`：未吸収差分が無いことを照合して閉鎖。
3. **滞留 worktree（206個・143 GB）の段階整理**
   - **第1波（即時安全）**：[棚卸し台帳 §2](simplification-inventory-2026-11-21.md#2-全-206-worktree-の実測分類総容量-143-gb) の `clean & merged` 24本について、`.orchestration/` や `docs/` からの生パス参照と未保存証跡が無いことを確認し、`git worktree remove` で削除する（これだけで `hr2-range-20261107` 26 GB を含む大幅削減）。
   - **第2波**：`clean & unmerged` 44本はブランチを残したまま worktree だけ削除する。
   - **第3波**：`dirty` 138本（稼働中を除く）はスクリプトで `git diff` を点検し、未吸収コード差分があるものだけ `archive/wt-<name>` ブランチへ退避 commit してから worktree を削除する。`git worktree prune` を実行。

**禁止事項**：取得済み科学証跡（WAV/TSV/JSON の唯一コピー）を確認なしに消さない。

**受入ゲート**：
- `git status`（`main`）がクリーン、かつ `HEAD == origin/main`。
- `git worktree list` が 5 本以内、`.worktrees/` の不要 `target/`（90 GB）が解消されていること。

---

### Unit S1：設計の引き算 —— 棚卸し確定とドキュメント分離（目安：11/25〜12/01）

**2026-11-21 完了（SIMP-S1A / S1B）**。監査と文書分離のうち、正本・棚卸し台帳・規約の統合 commit は `46b172c7d04663da7f2fdb8c5fe3cfd910d79411`。作者承認済みの分類 C は S2-a で実施し、分類 B の実装は S2-b で扱う。

**前提**：Unit S0 完了。[現行アーキテクチャ正本](../architecture-current.md) と [棚卸し台帳 §3](simplification-inventory-2026-11-21.md#3-rhai-api186-エントリと-configrs-opt-in-機能の-4分類原案stage-1) を入力とする。

**手順**：
1. **作者レビューによる 4 分類（A/B/C/D）の確定**
   - 棚卸し台帳 §3 の全項目（特に `Tuning` 25 未使用語、`Research` 9 語、`config.rs` の opt-in 12 項目）について、作者と最終仕分けを確定する。
   - マニフェスト依存の候補層・和声拡張（分類 D：14 語＋`dcc.coupling_strength`）はここでは削らず保留する。
2. **`AGENTS.md`（1,085 行 / 74 KB）のスリム化**
   - 毎回の作業で必ず守る規約・コマンド・不変条件・Air-Gap・根拠区分を本体（目安 250〜350 行）に残す。
   - 巨大な細部リスト（個別モジュールの割当内訳や履歴細則）を `docs/operations/engineering-reference.md` へ分離し、`AGENTS.md` からリンクする。
3. **歴史アーカイブの明示**
   - `docs/design-notes/README.md` の冒頭に [現行アーキテクチャ正本](../architecture-current.md) を最優先入口として据え、`technote-ledger`・`dcc-neurocognitive-hierarchy.md` は歴史アーカイブ／詳細仕様であることを明記する。

**受入ゲート**：
- 作者承認済みの「S2 削除リスト（C）」と「S2 隔離リスト（B）」が確定していること。
- `AGENTS.md` と `docs/architecture-current.md` が現行コードと矛盾しないこと。

---

### Unit S2：コードの引き算 —— 廃止削除・研究隔離・`#[cfg(test)]` 掃除（目安：12/01〜12/08）

**2026-11-21 完了（S2-a / S2-b / S2-c）**。`2fdb8f2bcff994560992a4f29edec8fb22035257` に承認済み削除（Rhai 13入口、`use_incoherent_power`、`birth_surrogate`）と聴覚前段のテスト支援分離を統合。続く SIMP-S2B1 / S2B2 により、`Cargo.toml` の `research = []` feature（CI ゲート付き）へ `temporal_cognition` 16サブモジュール、`body_metabolism`、`render_prototype`、`felt_beat` の計 51 ファイル（約 30,700 行）を隔離完了。統合 commit は `da83b565a61cdb132b0b0d51d8c528bc6966cdf7`。

**前提**：Unit S1 で確定したリスト（分類 B・C のうちマニフェスト非依存のみ）を入力とする。削除と隔離は別 commit に分ける。

**手順**：
1. **S2-a（廃止削除：分類 C）**
   - 確定した不要フラグ（例：`birth_surrogate`, `render_prototype`, `use_incoherent_power` 等、承認分のみ）と不要 Rhai 語を削除。
   - `just gen_rhai_defs` を実行し、`rhai-defs/conchordal.d.rhai` と日英 API リファレンスを更新。
2. **S2-b（研究隔離：分類 B）**
   - `RESEARCH_FNS` や研究専用 `temporal_*` 設定・アッセイ経路を、可能な範囲で専用モジュール／Cargo feature（例：`research`）の裏へ隔離し、通常演奏の `Voice`・`WorkerState`・`AppConfig` から不要分岐を外す。
3. **S2-c（`#[cfg(test)]` の本番構造体侵食の掃除）**
   - [棚卸し台帳 §4.3](simplification-inventory-2026-11-21.md#43-cfgtest-の本番侵食上位stage-2-対象) の上位ファイル（`auditory_frontend.rs`, `pitch_core.rs`, `oscillator_bank.rs`, `voice.rs`, `worker.rs`, `flora.rs`）から、本番構造体フィールドや関数中腹に埋め込まれた `#[cfg(test)]` をテスト専用モジュール／トレイトへ退避する。

**受入ゲート**：
- 主要サンプル（`samples/*.rhai` のうち廃止語を使わない全曲）＋デモ 2 曲の `conchordal-render` 出力が変更前後で **bit 一致**（`sha256sum` 照合）。
- `cargo clippy --all-targets -- -D warnings` 通過。
- `cargo check --no-default-features --features web` 通過。
- `just test` 実行・`test_status.txt` 記録（exit=0）。

---

### Unit S3：God Struct の分解（純粋リファクタリング、目安：12/08〜12/15）

**2026-11-21 完了（SIMP-S3A / S3B / S3C）**。`Landscape` の観測・和声拡張の分離は `8cd156e3ea025170c59226719a1cef9561555c38`、Voice・Flora・Community と `WorkerState` の整理は `1046989ade872b22b35d08c95cd051a2ac628d58`。

**前提**：Unit S2 完了。**機能変更・パラメータ変更は一切禁止。**

**手順**（1 構造体＝1 commit で細かく検証）：
1. **S3-a：`Landscape`（`src/core/landscape.rs`）の 3 層分離**
   - `AcousticField`（`space`, `roughness*`, `harmonicity*`, `consonance*`, `habituation*`, `subjective_intensity`, `nsgt_power`）
   - `EcologyViews`（`loo`, `flora_observations`, `tonal_observations`, `tonal_context`, `pitch_set*`, `spectral_history`）
   - `HarmonicExtensions`（`support*`, `harmonic_relations*`, `harmonic_recorder`, `harmonic_now_sample`, `harmonic_query_version`）
   - UI スナップショット生成（`src/runtime/ui_frame.rs`）で不要な観測束のクローンを避けられる形に整理。
2. **S3-b：`Voice`・`Flora`・`Community` のサブ構造体化と `RefCell` 解消**
   - `Voice`：`support_*` を `Option<VoiceSupportState>`、`harmonic_*` を `Option<VoiceHarmonicState>` へ集約。
   - `Flora`：`support_*`・`harmonic_*` を同様にサブ構造体化。
   - `Community` / `Flora` 内の `RefCell` スクラッチ類を、明示的な `&mut Scratch` 引数またはサブ構造体分離で解消。
3. **S3-c：`WorkerState`（`src/runtime/worker.rs`）の整理**
   - 再生ライフサイクル状態・UI 配信状態・解析/リスナー同期状態をサブ構造体へ分離。

**受入ゲート**：
- 全登録サンプル＋デモ曲の native PCM が S2 完了時点と **完全 bit 一致**。
- `profile-alloc`（または既存割当検証）で定常 hop の新規確保増加なし。
- `cargo clippy --all-targets -- -D warnings`、`web` feature check、`just test`（exit=0）。

---

### Unit S4：巨大ファイル分割と身体モジュール境界（目安：12/15〜12/22）

**2026-11-21 完了（S4-a）**。pitch core と Rhai 文書レジストリの分割は `e82f4f46c23e0a79cd676f0eb054bcf60ab3fe08`、Voice と Flora の分割は `acc45bb1e8bbf368cd71e92ebce48944311c95e4`。S4-b は同日、作業木で身体境界を整理し、代表4曲の native PCM 一致と担当分の検査通過を確認した。統合 commit は `77aaf829fa4c127884b7e90c8dabe6f5edd58fed`。

**2026-11-22 の全曲照合（SIMP-PCM-SWEEP）**。S2〜S4 の各単位は代表4曲で照合していた。受入ゲートの「全サンプル＋デモ曲」を閉じるため、S2 の前の `46b172c7` と最後の整理の commit `da83b565` を比べた。その次の `33060257` は意図して音を変えたので範囲に含めない。
- 対象は `samples/*.rhai` の18曲と `samples/demo/` の2曲。二つの版で台本は同じで、廃止語を使う曲は無かった。
- 条件は各単位と同じ。debug 版 `conchordal-render`、`--seed 21`、コメントだけの明示 TOML で終端まで描画した（出力はモノラル 16 bit）。
- 20曲すべてで WAV の `sha256sum` と `cmp` が一致した。一覧は `.orchestration/units/SIMP-PCM-SWEEP/compare.tsv`。

**前提**：Unit S3 完了。**音・挙動を変えない純粋な構造整理のみ**を実施する（励起契約変更や音色遺伝は混ぜない）。

**手順**：
1. **S4-a：巨大本番ファイルの分割**（本番コード 1 ファイル 1,000 行以下を目安）
   - `src/life/pitch_core.rs`（2,905 行）→ `pitch_core/`（候補生成・スコア評価・選択更新・テスト）
   - `src/life/voice.rs`（2,340 行）→ 責務別の `#[path]` サブモジュール追加（既存のフラット配置規則に従う）
   - `src/life/flora.rs`（2,134 行）→ バイオマス力学・レーン合成・拡張評価へ分割
   - `src/scripting/docs.rs`（2,355 行）→ オーナー別テーブルへ分割
2. **S4-b：身体（合成）モジュール境界の確立**（`docs/design-notes/timbre.md`「Core and body modules」）
   - `src/synth/` と `src/life/sound/` に跨る既存 3 身体（`Sine` / `Harmonic` / `Modal`、および Phase 2 採用分）を、生態ロジックの `kind` 分岐から切り離す統一インターフェースの背後へ移す。
   - 目標：新しい合成方式を追加するときに `life/` の候補・エネルギー・予測側を横断修正せずに済む状態を作る。

**受入ゲート**：
- 全サンプル＋デモ曲の native PCM が S3 完了時点と **完全 bit 一致**。
- `cargo clippy --all-targets -- -D warnings`、`web` feature check、`just test`（exit=0）。
- [現行アーキテクチャ正本](../architecture-current.md) のモジュール規模表と課題欄を更新。

---

## 2. 並行トラック：マニフェスト1周年改訂（Unit M1〜M5、〜2026-12-25）

対象：`web/content/manifesto.ja.md` および `web/content/manifesto.md`（英語版）。  
主線（S0〜S4）を止めず、週 1 回の作者対話で磨く。

| Unit | 期間 | 作業内容 | 成果物 |
|---|---|---|---|
| **M1** | 11/21〜11/28 | **論点と1年差分の確定**：下記の論点チェックリストを作者と詰め、「残す核」「書き換える前提」「technote へ逃がす現状注記」を決める。 | `docs/roadmap/manifesto-v2-notes.md` |
| **M2** | 11/29〜12/05 | **構成（節立て）の確定**：1周年版の章立てと各節の主張を箇条書きで確定する。 | 構成案（承認済み） |
| **M3** | 12/06〜12/12 | **日本語初稿の執筆**：思想宣言としての強度を取り戻し、現状報告書的な注記を排除する。 | `manifesto.ja.md` 草案 |
| **M4** | 12/13〜12/19 | **推敲・独立レビュー・英訳・台帳監査**：日本語稿を削ぎ落とし、英語版 `manifesto.md` を対訳作成。`technote-ledger.ja.md` §9.1 の公約表と矛盾が無いことを照合。 | 日英レビュー済み稿 |
| **M5** | 12/20〜12/25 | **最終確認・公開**：`web/` ビルド（`npm ci && npm run build`）確認後、**2026-12-25 に公開**。公開後、棚卸し台帳の **分類 D（保留14語等）** の採否へ接続する。 | 1周年改訂マニフェスト公開 |

- **M1**。2026-11-21 に論点整理を `46b172c7d04663da7f2fdb8c5fe3cfd910d79411` に収録。作者判断の完了日は未記録。
- **M2**。2026-11-21 に構成草案を作成。承認完了日と `origin/main` の統合 commit は未記録。
- **M3**。2026-11-21 に日本語初稿草案を作成。確定日と `origin/main` の統合 commit は未記録。

### M1（Week 1）論点チェックリスト（すぐ議論を始められる項目）
1. **「協和の引力」の先にある「動きの原理」**：R/H/C だけでは安定点に止まるという1年の実証を踏まえ、「離れて戻る力（順応・時間文脈・生態の代謝と交替・作曲者の介入）」をどう位置づけるか。
2. **「コーパス非依存」と「文化・経験（Layer 3）」の緊張**：生成AIの統計模倣を拒否しつつ、旋律的期待の大半が聴取経験に依存するという事実をどう正直に言語化するか（第一原理／作曲者が与える文化・生態の性向／聴き手履歴の線引き）。
3. **「響き（身体・テクスチャ・空間）」の格上げ**：周波数軸・時間軸の二軸図式から、身体の共鳴・統計的テクスチャ（Flora/Rain）・共有音響空間（部屋）をどう組み込むか。
4. **作曲者の役割の再定義**：音符の配置者ではなく「環境条件・住人の性向・時間の大弧を設計する演出家（Author rule の担い手）」であることを、敗北ではなく楽器の積極的な形として書き切るか。
5. **射程と文体の整理**：時間軸節の「2026年9月時点の実装では…」のような現状注記を technote へ移し、生体信号閉ループや他芸術領域への射程を「遠い地平」として適切な重みに整えるか。

---

## 3. 実行分担：`codex sol6.2`（worker）vs 自分（統括）

| 段階 | 自分（統括）が直接やること | `codex sol6.2` を worker として進めること |
|---|---|---|
| **判断軸** | 設計境界の確定、作者との対話・採否確認、差分レビュー、`git` 破壊的操作の最終判断、マニフェスト執筆 | 範囲と受入ゲート（bit 一致・テスト・clippy・web check）が明確な**網羅監査・文書構造分離・純粋リファクタリング・確定済み削除/隔離** |
| **Stage 0（足場固め）** | ローカル `main` 同期、`body-phase2`（DR-39/41/42）試聴取りまとめ、worktree 削除の最終実行 | 206 worktree の参照・証跡チェック表作成（読み取り専用） |
| **Stage 1（設計の引き算）** | 4分類（A/B/C/D）の最終仕分け判断（作者と合意）、分割後 `AGENTS.md` のレビュー | **SIMP-S1A**：`Tuning` 25語・`Research` 9語・`config.rs` opt-in 12項目の全参照元・削減行数監査<br>**SIMP-S1B**：`AGENTS.md`（1,085行）の「必読規約本体＋詳細リファレンス」分離と `docs/design-notes/README.md` 入口整理 |
| **Stage 2（コードの引き算）** | 削除・隔離の最終承認、差分レビュー | **SIMP-S2C（先行可）**：`src/core/auditory_frontend.rs`・`src/runtime/worker.rs` 等の `#[cfg(test)]` 本番侵食掃除<br>**SIMP-S2A / S2B**：S1 確定後の廃止コード削除・研究隔離・`just gen_rhai_defs` |
| **Stage 3（God Struct 分解）** | 分解後の型境界レビュー、不変条件確認 | **SIMP-S3A / S3B / S3C**：`Landscape` 3層分離、`Voice`/`Flora`/`Community` サブ構造体化＆`RefCell`解消、`WorkerState` 整理（bit 一致検証込み） |
| **Stage 4（分割・身体境界）** | 身体モジュール境界の設計承認、最終レビュー | **SIMP-S4A**：`pitch_core.rs` / `voice.rs` / `flora.rs` / `docs.rs` の機械的分割<br>**SIMP-S4B**：既存身体の新境界移行＋適合テスト |
| **並行（マニフェスト M1〜M5）** | **全工程を主導**（論点整理・執筆・推敲・作者対話） | **M4 補助のみ**：英語版対訳チェック、`technote-ledger.ja.md` §9.1 公約表との機械照合、`web/` ビルド検証 |

---

## 4. 各 Unit 開始時の運用ルール（統括・エージェント向け）

1. 各 Unit 開始前に、本指示書の該当節と [棚卸し台帳](simplification-inventory-2026-11-21.md) を読み、対象範囲と非対象範囲（やらないこと）を1画面に宣言してから着手する。
2. 1 Unit の中で「仕様変更」と「bit 一致リファクタ」を混在させない。
3. 各 Unit 終了時に、削減行数・削減 API 数・bit 一致照合・`test_status.txt` を記録し、本指示書の該当項に完了日と commit hash を追記する。
