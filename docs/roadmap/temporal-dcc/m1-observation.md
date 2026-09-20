# M1の初期観測経路

日付: 2026-09-12。状態: W3/M1の入力・診断を実装。M1全体、認知モデル、M0全体の完了ではない。
前段の数値準備は[m0-rust-preparation.md](m0-rust-preparation.md)、今回のsourceと検証結果は
[m1-observation.json](m1-observation.json)に保存する。

## 実行できる操作

```rhai
temporal_mode("observe");
let voice = place(sine().sustain().anchor().amp(0.08), at(220.0));
wait(1.0);
release(voice);
wait(0.2);
```

`temporal_mode`はScenarioの開始前設定で、省略時と`"off"`は既存動作を選ぶ。
top-levelで、演奏eventと時刻進行より前に一度だけ宣言できる。同じ値の重複、sectionや
parallelのclosure、script関数内、時刻進行後、未知のmodeは拒否する。
`"participate"`は未実装として拒否する。disposition APIはまだない。

`observe`はhabitatとpresentationの既存NSGT出力を、それぞれ独立した観測workerへ渡す。
機器なしの観測runはreport有無によらず決定的な配送を使い、profileにも実際のlistener起動を記録する。
画面とreportの有無にかかわらず両busを観測する。入力にVoice ID、作者のsectionラベル、
予定された音列、私有状態を渡さない。追加した観測から生成への経路はなく、関係・帰結・行動は
未実装である。順序照合の数値部品は引き続きtest buildだけで使い、実音との対応付けを行わない。

## 時刻・欠落・容量

tapは`analysis_worker`のNSGT処理直後、generator向け結果の間引き前に位置する。
sample時刻は元のframe IDから計算する。支持区間はNSGTの全窓で保守的に扱い、初回と
原音欠落後は窓が埋まるまでpowerをunknownにする。窓全体を観測した無音は既知ゼロになる。
原音のframe IDが飛んだ場合はsource epochを進め、欠落sample数を累積する。
新しいrunは別workerと空状態から始まる。ライブのmode・解析寸法変更は追加していない。

各busのqueueは64件。オフラインの決定的な実行ではbackpressure、実時間のtapは`try_send`を使い、
飽和時は配送欠落数を記録する。原音欠落と配送欠落は別項目である。後続のモデルは配送欠落を
連続した認知支持として扱ってはならない。現在は関係を生成しないため、この復旧は未実装である。
sample上の利用可能時刻と、queue内の壁時計遅延を別々に記録し、遅延を新しい音響支持へ読み替えない。
元のframe順序・版・bus・時刻整合・有限の非負powerを検査する。まだfitや保存codecの読み込みがないため、
今回の旧版拒否だけをMR3全体の合格とはしない。

終了時は解析入力を閉じて未処理入力を排出し、observerの終了を待って最終reportを保存する。
GUIの終了画面にも同じ要約を再送する。停止した生成の後でのみ待ち、音声callbackからjoinしない。
各hopは固定長snapshotを`try_lock`で読む。追加のheap確保は解析threadの各配送につき一つの
`Arc<[f32]>`コピーとqueue管理であり、主workerのhopでは追加しない。これらの費用と追加した
presentation解析を含む全O04は未計測である。

## 検証と残る範囲

- 既存NSGTで24 hopを処理し、間引き後の配送が一件でもobserverは24件、全窓支持は21件を受信する。
- 初回・原音欠落後のwarmup、既知ゼロ、原時刻、キュー飽和時とEOFの欠落数を検査する。
- 不整合な版・bus・epoch・順序・時刻・非有限powerが採用済み証拠を進めないことを検査する。
- `off`／`observe`とreport有無の4条件でWAVをbyte比較する。presentationだけを鳴らし、habitatの
  既知ゼロとpresentationの非ゼロを同時に確認する。両busのEOFまで全frame数と元支持を照合する。
- Rhai宣言境界、生成API参照の同期、通常buildと全Rustテストを検査する。

最終Rust検査は766件通過・13件ignore・失敗0。`cargo clippy -- -D warnings`、
`cargo check --all-targets`、fmt、生成API同期、日英mdBookとZolaのbuildが通過した。
全targetのClippyには以前からの二分類13件が残るため、通常Clippyの合格と区別する。
二分類を許容した全target検査では他の警告はない。
保存した4条件レンダーは同一WAVとなり、両busとも58 frameを受信した。全窓支持27件、
欠落・拒否ゼロ、最終支持は`[13312, 29696)` sampleである。これは短いオフラインfixtureの結果であり、
O04が求める600秒・64 Voice・全数値処理の結果ではない。

GUIの描画内容は同じsnapshotから組み立てるが、実画面の操作・見切れ、実時間音声機器、長時間の
遅延と資源上限は今回の自動検査に含まない。遅延した認知結果の期限拒否、ridge/grouping、
候補とbank、関係記憶、帰結表、private trace、fit／保存形式を含むMR3は後続に残る。
M0の全負荷・全特徴とfit表・実音回収・課題素材・分割・pilot・機関手続きも未達のまま保持する。
