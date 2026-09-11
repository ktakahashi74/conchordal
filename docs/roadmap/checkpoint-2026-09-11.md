# 時間構造DCCの停止地点

日付: 2026-09-11
状態: M0の研究実装・数値検査・設計改訂と、Rustを実装正本にする計画修正案を保存する。
M0は未完了。修正案は未適用であり、新計画による本体実装は開始していない。

## 今回保存する成果

前回の`ec67b50`以後の変更を保存する。`src/`の楽器本体と和声のR/H/Cは変更していない。

| 範囲 | 保存する成果 | 残る条件 |
|---|---|---|
| M0の登録 | 基準・機器・特徴・課題・対照・回収・資源のJSON記録と[M0台帳](temporal-dcc/m0.md) | 全入力と適合手順、実音尺度、素材・収集条件等の確定 |
| 数値参照 | 音響accent、記述子、順序照合、記憶保持、bank、queue、payload、consumer、section等のPython参照と検査 | 実音からの全帰属・推論・有界配送と本体への接続 |
| 回収実験 | episode・私有traceの失敗と改訂、条件付き回収結果、再生成・集計用スクリプト | 実際の身体・波形・接触機会と人的実現可能性。合成入力の回収を実音へ自動転用しない |
| Rust数値照合器 | Python参照と比較する独立native kernel、packed入力、検査・性能記録 | dispatch／receiptを含む実経路、consumerの遅延、両bus・64 Voiceの全負荷O04 |
| 原理と文書 | 日英Manifestoの実装／設計目標の区別、神経認知メモ・完成計画・technoteの改訂 | 新たな外部レビュー、認知課題・音楽的な採用は未実施 |
| モデル交換 | [交換単位・共通契約・再適合とMR1–MR3](temporal-dcc/model-replacement.md) | 比較手順の登録まで。交換実験は未実施 |
| 工程修正案 | [Rustを実装正本にする修正案](temporal-dcc/implementation-plan-revision.md) | 正本へ未適用。最初の数値移行とMR1を通してから受動経路へ接続する案 |

現在の数値参照のfeature版は`m0-registration-29`。過去の外部レビュー・実験結果は
それぞれに保存したsourceと仕様の版にだけ対応し、このcommitの全内容の検証を意味しない。
登録JSON内の過去のsource hashを、現在のsourceへ付け替えない。

進行中のpacked scheduler／worker接続も保存する。
`scripts/evaluate_temporal_matcher_worker.py`の`NativeQueryWorker`には、専用の全経路検査がまだない。
既存テストの実行と、新しいthread dispatch／receipt・実時間性能の検証を区別する。
この接続の完成やO04の通過を、今回のcheckpointから推定しない。

## 修正案の位置づけと再開

修正案は、今後の数値実装をRustへ集約し、Pythonを独立参照・統計等へ限定する。
M0の実装準備と収集開始条件を分け、本体の受動経路を早く接続する。
M0–M9とO01–O22の義務、T1–T7の四判定、階層間の相互作用と全曲の完成範囲は維持する。

会話で示した残作業の約1〜3割削減は、重複作業の割合と移行費用を仮定した試算である。
削減実績や確定トークン数ではなく、過去の消費量・人の収集と試聴時間は含めない。

次に修正案を採用する場合は、そのW0で正本・担当・停止条件を同期し、
W1のRust照合部品、W2のMR1、W3の実音による受動経路へ進む。
採用前は[現行完成計画](temporal-dcc-completion.md)を正本とし、
この保存をM0完了・修正案採用・人の収集開始の承認とは扱わない。

## 保存範囲と検証の限界

文書、研究用コード、テスト、登録JSONをcommit対象とする。
`.claude/`と`.codex`はローカル設定として対象外とする。
`target/`内の実験出力・音源・native binaryとignoredの実行ログはローカルに残り、commitには含まれない。
別環境で過去の実験を検証するには、記録が指す成果物の保存または再生成も必要になる。
楽器への録音経路は追加せず、音源を保存する実験は分離rendererを使う。

停止前の検査ログは`test_report.txt`、`test_status.txt`、
`target/temporal-dcc/checkpoint-20260911/`へ保存する。
これらはローカルの検査であり、新しい聴取実験・作者試聴・実機全負荷検証の結果ではない。

停止前に次の検査を通した。

- `cargo clippy -- -D warnings`、`cargo fmt --all -- --check`、`cargo check --all-targets`。
- `RUST_BACKTRACE=1 cargo test -- --nocapture`: 746件通過、0件失敗、12件ignored。全文と終了値0を上記ログへ保存した。
- `/tmp/conchordal-context-scores-venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`:
  871件通過（33.412秒）。native照合器の43件も含む。BLAS／OMPのthread数は各1とした。
- 登録JSON 47件のparse、追加Python 41ファイルの構文検査、日英サイトのZola build。
- `git diff --check`。楽器本体の`src/`とCargo設定が前回commitから変更されていないことも確認した。
