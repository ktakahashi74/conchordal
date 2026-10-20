# 情報地形（予測精度と驚き）の審査・登録・実測記録

Flora の暫定の力学が静止した和音に縮退した件を受けて、聴き手側のスペクトル予測に関わる知覚量を DCC 側に置けるかを、機構選択規則に照らして吟味した記録である。経過、結果、作者の判断は [台帳 §9.3.56](../../design-notes/technote-ledger.ja.md#info-terrain)（英語版は [technote-ledger.md](../../design-notes/technote-ledger.md#info-terrain)）にまとめた。このディレクトリは、その根拠となる文書と結果の表を保存する。

## 文書

- [scrutiny.ja.md](scrutiny.ja.md)：機構選択規則による審査（機能 A：受動的な観測、機能 B：生成への結合）。
- [registration-a.ja.md](registration-a.ja.md)：登録 A（原型版と拡張版の驚きと精度）。実測は A-1 と呼ぶ。
- [registration-a2.ja.md](registration-a2.ja.md)：登録 A-2（知覚の床と拡張版の起動）。A-1 を置き換える。
- [registration-a3.ja.md](registration-a3.ja.md)：登録 A-3（EFE に着想を得た受動的な合成値）。第1段で不合格となり、採用しなかった。

いずれも `target/info-terrain-20261020/` から 2026-10-21 に移した。本文は書き換えず、冒頭に記録の注記だけを加えた。

## 結果の表

評価ハーネスが出力した Markdown と、各段の検証記録 `verification.json` を、そのまま写した。中の絶対パスと `target/` のパスは、取得時の場所を指す。

- `results/a1/`：A-1 の統制刺激、render、外部照合、WAV の単体実行。
- `results/a2/`：A-2 の同じ一式と、A-1 との比較（`a1_vs_a2.md`）。
- `results/exploratory_dimensions.md`：登録していない探索 A4。複合指標の登録の動機にだけ使った。
- `results/a3/`：A-3 の第1段（`stage1.md`）と、第2段を実行しなかった記録（`stage2.md`）。

## 保存していないもの

`verification.json` 以外の JSON、trace、統制刺激と render の WAV、Anikin 2026 の刺激、独立レビューの本文とログ、worker の記録は `target/info-terrain-20261020/` に残したままで、ここには写していない。`target/` は消えうる。

評価ハーネス（`src/core/info_terrain/info_terrain_eval/`、`src/core/info_terrain/efe/evaluation.rs`。crate 内の試験で、既定では実行しない）は、`target/info-terrain-20261020/` から入力を読み、そこへ結果を書く。

- A-2 の一式（`info_terrain_eval_registered_suite`）は `results/` に書く。A-1 との比較のため、保存済みの `results-a1/` も読む。A-1 のコードは A-2 で置き換えたので、A-1 の値は今のコードからは再生成できない。
- 外部照合には、OSF bgzvc の `stimuli.zip` を復号した WAV（`anikin-stimuli-wav/`）と `fit_per_sound.csv` を、同じ場所に置く必要がある（手順は登録 A §7.4）。
- A-3（`efe_registered_two_stage_suite`）は `results-a3/` に書く。第1段は合成した系列だけを使う。ただし出所の記録のため、登録 3 本を `target/info-terrain-20261020/` から読み、表題の行を確かめる。`target/` が消えていたら、このディレクトリの登録を写してから実行する（表題の行は変えていないので、確認を通る）。

```sh
cargo test --release --lib info_terrain_eval -- --ignored --nocapture
cargo test --release --lib efe_registered_two_stage_suite -- --ignored --nocapture
```
