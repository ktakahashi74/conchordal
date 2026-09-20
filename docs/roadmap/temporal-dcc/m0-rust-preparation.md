# M0 Rust数値準備とMR1

日付: 2026-09-12。状態: W0–W2の技術実装。M0全体は未完了。
本記録のsource hash・build状態・資源値はW0–W2保存時点に対応する。現在の実音接続は
[M1初期観測](m1-observation.md)へ進んだ。旧manifestのsource／文書は保存archiveと照合し、
現在のcheckoutの検証済みhashとして使わない。
正本は[完成計画](../temporal-dcc-completion.md)。[工程改訂](implementation-plan-revision.md)を
今回のM0実装指示に基づいて適用した。数値・source・入力・結果の対応は
[manifest](m0-rust-preparation.json)、資源の実測値は[m0-rust-resources.json](m0-rust-resources.json)に保存する。

## 実装した範囲

| 単位 | 成果物 | 検証の範囲 |
|---|---|---|
| W0 | 工程・M/O担当と停止条件の同期、[操作・表示・試聴契約](author-interface-contract.md)、source／fixture一覧 | 実装準備と収集条件を分けた。構文案の所有者・既定・適用時点・エラー・未実装状態を記載。身体／文脈／交互作用の追加要件も保持 |
| W1 | `src/temporal_cognition/matcher.rs`と`memory.rs`の順序付き照合 | frozen Pythonの48組のDTW／anchor、6組のqueryと候補順位・f64 bit・経路、小型81入力の全経路列挙、powと丸めの既存反例、mask・時刻・容量・尺度拒否 |
| W2/MR1 | 同じ観測を使う`memory_orderless`と`transport.rs`の共通controller／契約検査用consumer | 手計算とPython標準統計の12組、全区間bagと並べ替え、既知ゼロ／欠測、元支持・受領時刻、別bus／epoch／世代／モデル、遅延、退役・再利用禁止、原子的な受領、二重受領・容量・待ち状態 |

入口はcrate内部の`src/temporal_cognition.rs`。M1のruntime接続前なので、現在は`src/lib.rs`の
`cfg(test)`からコンパイルし、通常の楽器binaryへ未使用経路を組み込まない。実験・検査はこのRust
実装を直接呼ぶ。M1ではこの入口を通常buildへ移して実際の入出力へ接続する。公開module・汎用trait・
新しいcrate・作者向けの研究モデル選択APIは追加していない。

旧`temporal_matcher_kernel.rs`のf64数値算法を安全なsliceと所有へ移した。CPython API、ctypes、
GIL、Python辞書と例外順序は移していない。Rust側は`InvalidInput`と`NumericalRange`を返す。
有限範囲を超える演算は拒否し、Pythonへのfallbackや非有限診断の採用はしない。この入力域の区別を
Python全入力の完全な互換性とは呼ばない。途中で失敗したscratchは結果として公開しない。

128 knot、256 episode、32 coarse anchor、16候補、最大4 refinement、band半径16を維持した。
原時刻でbandを引き、同値のDP操作は対角・挿入・削除、endpointは早い位置を優先する。
候補cutoffの同値とband端を曖昧さとして残す。f32化・容量削減・新しいfitは行っていない。
今回のquery adapterは登録済みepisodeの借用sliceを受ける。完全なbankの保持・干渉・codec・
endpoint ledger・schedulerを移植したものではない。

順序なし対照は全区間の有効値の平均・母標準偏差を使う。timestampと経路はcostへ渡さず、
局所intervalは並べ替え前に保存した別配列から読む。現在の数値Knotにはinterval保存欄がないため、
MR1 fixtureのadapterで別配列を作る。実音producerとcodecへの接続はM3に残る。
bagの加算は値順に固定し、同じbagの並べ替えで丸めが変わらないようにした。
標準統計参照との許容差は`8e-15 * max(1, abs(cost))`、同じbagの並べ替えはbit一致とする。
これは順序なし対照の新規数値実装の検査規約であり、順序付きf64 bit一致の緩和ではない。

共通controllerはモデル名を分岐せず、版token、bus、epoch、query・支持ID、元の標本時刻、
期限と生存handleだけを扱う。一bus一件のpending ticket、最大256 handle、最大64関係を保持する。
全packetの検査後にだけticketを消費する。退役handleを同epoch内で再登録しない。
出力にraw DTW cost・経路を渡さず、支持・曖昧さ・同定済み変形を渡す。costと経路はモデル内の
検査reportに残る。未校正の支持候補であり、共通確率や生成命令ではない。

## 数値結果と資源

MR1の小型入力はcue `[0,1,0,-1]`、episode 1は同じ順序、episode 2は`[0,-1,0,1]`。
順序付き照合は同一列と並べ替えを区別し、順序なし対照は両者を同値にする。
実際のcost・経路・変形・受領数・時刻は資源JSONの`mr1_exchange`に保存した。
支持不足をゼロcostの支持へ変更せず、遅延・旧世代等の誤採用はfixtureでゼロを要求した。
これは構成した入力での交換境界の確認であり、T4/T6の認知的な対照採用ではない。

release計測は256 episode×128 knot、cue128、1／10有効座標、二方式、各20回warmup＋200回測定。
各呼出しの入力は同じ借用bankで、query内の検証・coarse・refinement・DTW・report構築を含む。
bank／入力の生成は計時外。MR1の配送測定ではpacketのcloneを実際に一回行う。
固定配列とpacketの物理サイズを記録し、モデルのreport／経路Vecの割当を別の残条件とする。
allocator／RSS／全stackの完全な実測ではなく、Python旧版と同一入力による速度比も主張しない。
この有限な単一caller計測から、両worker・実機・64 VoiceのO04を合格にしない。

## 再現

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/generate_temporal_rust_fixtures.py --output /tmp/m0-reference.json
cmp tests/fixtures/temporal_cognition/reference.json /tmp/m0-reference.json
cargo test --lib temporal_cognition
CONCHORDAL_M0_REPORT=/tmp/m0-rust-resources-new.json cargo test --release --lib temporal_cognition::tests::resource_assay -- --ignored --exact --nocapture
```

資源出力は既存fileへの上書きを拒否する。fixtureは生成時のsource hashを持ち、Rustを呼んで
期待値を作らない。旧Python workerの未検証接続を再利用・昇格せず、過去の数値source・結果は保持した。
リポジトリ全体のCargo test全文と同一shellの終了値は`test_report.txt`／`test_status.txt`、
検査状況と制限はmanifestへ記録する。テスト件数の増加だけを認知的な証拠にしない。

## 残条件と次の実装

W0–W2からM1の技術実装へ進む準備は成立した。M1では既存analysisの順序付き記録、二busの独立状態、
gap／epoch／EOF／遅延／順序逆転、GUI・report独立性、unknownと表示理由、最初の実利用先でのMR3を
検査する。初期の受動経路ができても、全認知modeが完成したとは扱わない。

M0にはO04の全処理・copy・queue・allocator・機器・両worker・64 Voiceの非ゼロ同時負荷、
全特徴・提案・対照・fit表、実音尺度での回収・event-head予測安定性、素材・分割・機関手続き・
言語pilotと人の収集条件が残る。O05の外部判断を数値成功で補わない。
身体アフォーダンスに関する同時追記は保持したが、このMR1で身体／文脈介入を実施したわけではない。
M2–M5の技術実装で必要な実入力・処理を作り、M6以降の採用とは区別してM0残条件を検査する。
