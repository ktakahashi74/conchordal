# I11 第1段の登録入力と取得・判定 script

[I11-1の登録](../i11-onset-comparison.md)の検査を、どのマシンからでも実行できるようにした入力一式である。
2026-09-23 までこれらは `target/i11-stage1-20260922/`（git 管理外）にしか無く、別マシンからは
登録12条件を回せなかった。入力と script だけをここへ移し、**出力（WAV・report・profile・log）は
一切置かない**。出力は従来どおり `target/<取得ごとのディレクトリ>/` へ書く。

## 中身

| もの | 説明 |
| --- | --- |
| `inputs/*.rhai` | 登録12条件のscenario。`sine`／`harmonic`／`modal` × `hold`／`flow` × 4／16／64 Voice。I10の入力のバイト複写 |
| `inputs/config-{none,body,proxy}.toml` | §4.10の3つの設定variant。`none` は `[temporal_onset_comparison]` 節を持たない（`None`）、`body` と `proxy` は `footprint` を持つ |
| `plan.json` | 登録。範囲・12条件・variant・mode・指標・許容値、判定基準の原文と英訳、`inputs_sha256`、基準commit |
| `registration.json` | §5.9のoffline再投影用。`src/runtime/body_profiles/representative_gap.rs` の ignored test が読む |
| `acquire.py` | 取得。12条件 × variant × （render／instrument report有無） |
| `api_effect.py` | §5.5の判定（`body` 対 `proxy`、`body` 対 `none`） |
| `scripts/locate_i11_first_divergence.py`（リポジトリ直下の `scripts/`） | §5.5(a)。`body` と `proxy` のreport対で、時刻順に最初に選択が分かれた決定を取り出し、`power_k` 以外に違う入力があれば列挙する |
| `hop_path.py` | §5.7の統計と元の許容による判定 |
| `aa_floor_check.py` | §5.7をA/A floorの規則で判定する（floorのJSONを `--floor` で渡す） |
| `own_us_check.py` | §5.7の`elapsed_us`を`own_us = elapsed_us - analysis_wait_us - listener_wait_us`で判定する。`floor`でA/A passからfloorを作り、`judge`で判定する |
| `bit_identity.py` | §5.4a（`None` が基準commitとbit一致するか） |
| `scripts/compare_i11_candidate_records.py`（リポジトリ直下の `scripts/`） | §5.4b。基準版と新版の `None` のreport対で、候補record（`body_candidate_energy`）を突き合わせ、`processing_us` 以外の全欄をbit比較する |
| §5.9の取得と集計 | ここには置かない。取得は `src/runtime/body_profiles/representative_gap.rs` の ignored test `acquire_representative_gap`（`registration.json` を読む）、集計は `scripts/summarize_i11_representative_gap.py`。下の「§5.9」の手順を使う |
| `register.py` | 入力を凍結して `plan.json` を書いた script。履歴として置く。再実行しない |

`inputs_sha256` の**値は2026-09-22に凍結したものそのまま**で、キーだけを新しい場所へ向け直した
（`plan.json` の `inputs_location` に経緯がある）。複写がこの値に一致することを確認済みで、以後も
`acquire.py` が実行前に全件検証する。**入力は書き換えない。** 変更するなら新しい登録として起こす。

## 実行

出力先は毎回新しいディレクトリにする。`--root` が出力先、`--plan` が登録の場所である。

```sh
D=docs/roadmap/temporal-dcc/i11-inputs
OUT=target/i11-stage1-$(date +%Y%m%d)

cargo build --release
mkdir -p "$OUT"

# 取得（例: §5.7のbody対none、登録8条件、report有無、交互3反復）
python3 $D/acquire.py --root "$OUT" --plan $D/plan.json \
  --mode report --mode no-report --variant body --variant none \
  --case sine-hold-4 --case sine-hold-16 --case sine-flow-4 --case sine-flow-16 \
  --case harmonic-flow-4 --case harmonic-flow-16 --case modal-flow-4 --case modal-flow-16

# 判定
python3 $D/hop_path.py   --root "$OUT" --plan $D/plan.json
python3 $D/api_effect.py --root "$OUT" --plan $D/plan.json
python3 $D/aa_floor_check.py --root "$OUT" --floor <A/A floorのJSON>
```

`--inputs` を省くと `--plan` と同じディレクトリの `inputs/` を使う。`acquire.py` が検証するのは
`plan.json` の `inputs_sha256` に載っている場所なので、`--inputs` で別の場所を指した場合は
検証した先と実行した先が食い違う。既定のまま使うこと。

### §5.9（代表条件と実条件の差）

取得は `src/runtime/body_profiles/representative_gap.rs` の ignored test で、`registration.json` の12条件を
読む。入出力は環境変数で渡し、**出力ディレクトリは存在しないこと**が要る（test が自分で作る）。集計は
`scripts/summarize_i11_representative_gap.py` で、`gap.jsonl` と `report.jsonl` を突き合わせて
`power_gap`・`overlap_gap`・`term_gap` の分布をfootprint源別に出す。合否は付けない（分布の報告）。

```sh
OUT=target/i11-gap-$(date +%Y%m%d)
CONCHORDAL_I11_GAP_INPUTS=$PWD/docs/roadmap/temporal-dcc/i11-inputs \
CONCHORDAL_I11_GAP_OUTPUT=$PWD/$OUT \
  cargo test --lib acquire_representative_gap -- --ignored --nocapture
python3 scripts/summarize_i11_representative_gap.py "$OUT" --output "$OUT/summary.json"
```

`[profile.dev]` は `opt-level = 1` なので、12条件の取得は `--release` なしで約3分である（2026-09-23 実測。
出力は約780 MB）。

## script に加えた変更（2026-09-23、ロジックは変えていない）

`target/<dir>/` に置かれている前提のパス処理だけを直した。測定と判定の中身は触っていない。

- リポジトリ根を `Cargo.toml` を上へ探して決める（旧: script の2つ上のディレクトリ）。`acquire.py`・`register.py`
- `--plan` を追加（既定 `<root>/plan.json`）。`acquire.py`・`api_effect.py`・`hop_path.py`・`bit_identity.py`
- `--inputs` を追加（既定 `<plan のディレクトリ>/inputs`）。`acquire.py`
- `--floor` を必須に、`--hop-path` の既定を `<root>/hop-path.json` に。`aa_floor_check.py`

変更後、既存の取得データに対して `hop_path.py`・`api_effect.py`・`aa_floor_check.py` を走らせ、移設前の
出力と一致することを確認した（`hop_path.py` は `plan.json` の `cases` が12条件に戻るため64 Voiceの4条件が
増えるが、共通する8条件の中身は完全に一致する）。
