# Test fixtures

`temporal_cognition/*.json` は、Python の独立参照と構成した入力から生成した Rust test 用データである。実録音からの知覚・人の判断・科学的受入全体を保証するデータではない。各 JSON の schema と scope、および対応する Rust test の条件を保持する。

## Temporal cognition の生成器

リポジトリ直下で、固定した Python 環境から実行する。依存の導入手順は [scripts/README.md](../../scripts/README.md) を参照する。

| 出力 (`tests/fixtures/temporal_cognition/`) | 生成コマンド | 主な参照 |
| --- | --- | --- |
| `accent-ledger.json` | `python scripts/generate_temporal_accent_ledger_fixtures.py` | `temporal_accent_reference.py` の有限配送列 |
| `features.json` | `python scripts/generate_temporal_feature_fixtures.py` | accent と raw feature の独立参照 |
| `trajectories.json` | `python scripts/generate_temporal_trajectory_fixtures.py` | 有理数の peak・所有判定 |
| `ridges.json` | `python scripts/generate_temporal_ridge_fixtures.py` | Decimal の residual・secant |
| `groups.json` | `python scripts/generate_temporal_group_fixtures.py` | Decimal の割当・保存 member 最大値 |
| `grouping.json` | `python scripts/generate_temporal_grouping_fixtures.py` | Decimal Pearson・complete-link |
| `proposals.json` | `python scripts/generate_temporal_proposal_fixtures.py` | 有理数の重み・競合判定 |
| `producer.json` | `python scripts/generate_temporal_producer_fixtures.py` | 不変 key と辞書・集合による producer |
| `lifecycle.json` | `python scripts/generate_temporal_lifecycle_fixtures.py` | slot・clock・admission |
| `periods.json` | `python scripts/generate_temporal_period_fixtures.py` | 全 bin の kernel と非 cache 加算 |

生成器の seed・精度・入力列は source 内で固定されている。再生成は意図した参照更新の際に行い、上書き前の JSON を保持し、生成後に schema、入力、expected、source hash の差分を確認する。生成器と fixture の更新時点が異なる場合もあるため、現在の生成器による byte 一致を無条件には仮定しない。Rust の対応 test は `src/temporal_cognition/` にある。通常の test 実行は fixture を読み、再生成しない。

## その他

`ppm_decay_reference.json` は PPM v0.1.1 由来の保存済み予測ベクトルであり、上の生成器群の出力ではない。`scripts/tests/test_evaluate_ppm_decay_reference.py` が Python 参照と照合する。`temporal_meter_t2*` と `articulation_persistence/` は Rust の描画・統合検査が読む固定シナリオと設定である。
