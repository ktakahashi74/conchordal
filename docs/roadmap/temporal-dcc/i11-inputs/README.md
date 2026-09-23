# I11 第1段の登録入力と取得・判定 script

[I11-1の登録](../i11-onset-comparison.md)の検査を、どのマシンからでも実行できるようにした入力一式である。
2026-09-23 までこれらは `target/i11-stage1-20260922/`（git 管理外）にしか無く、別マシンからは
登録12条件を回せなかった。入力と script だけをここへ移し、**出力（WAV・report・profile・log）は
一切置かない**。出力は従来どおり `target/<取得ごとのディレクトリ>/` へ書く。

## 中身

| もの | 説明 |
| --- | --- |
| `inputs/*.rhai` | 登録12条件のscenario。`sine-hold` 3条件と、`sine-flow`／`harmonic-flow`／`modal-flow` 各3条件（4／16／64 Voice）。I10の入力のバイト複写 |
| `inputs/config-{none,body,proxy}.toml` | §4.10の3つの設定variant。`none` は `[temporal_onset_comparison]` 節を持たない（`None`）、`body` と `proxy` は `footprint` を持つ |
| `plan.json` | 登録。範囲・12条件・variant・mode・指標・許容値、判定基準の原文と英訳、`inputs_sha256`、基準commit |
| `registration.json` | §5.9のoffline再投影用。`src/runtime/body_profiles/representative_gap.rs` の ignored test が読む |
| `acquire.py` | 取得。12条件 × variant × （render／instrument report有無） |
| `api_effect.py` | §5.5の判定（`body` 対 `proxy`、`body` 対 `none`） |
| `scripts/locate_i11_first_divergence.py`（リポジトリ直下の `scripts/`） | §5.5(a)。全決定を時刻順に照合し、最初に選択またはskipが分かれるまでの非power入力、候補構造、同じrecord識別子を厳密比較する。終了コード0だけでは素材ごとの分岐要件を満たしたとは判定しない |
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
§5.7の最後の判定には、取得済みA/A floor JSONの場所を`AA_FLOOR_JSON`に設定する。

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
python3 $D/aa_floor_check.py --root "$OUT" --floor "$AA_FLOOR_JSON"
```

`--inputs` を省くと `--plan` と同じディレクトリの `inputs/` を使う。`acquire.py` が検証するのは
`plan.json` の `inputs_sha256` に載っている場所なので、`--inputs` で別の場所を指した場合は
検証した先と実行した先が食い違う。既定のまま使うこと。

### 同一record修正後の§5.5(a)再取得（blanche）

`body` と `proxy` の両方が代表身体recordを要求する修正後のbinaryを使う。登録済み入力とconfigは
書き換えない。3つのconfigは同じ絶対パス
`/home/shafi/lwrk/conchordal/target/i10-action-profiles-20260918.bin`を参照する。
このprofileのSHA-256は`033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5`。
このMacにはprofileと基準取得が無いので、ここでは実走できない。以下は両方が存在するblancheで実行する。
基準取得は`target/i11-stage1-54b-20260923/baseline/`を使い、その中の`render` binaryのSHA-256を
`plan.json`に登録された`3614d998e07c04c7af1913f97f515cb0edf0be4e2aec6b18331703e67fa58b5e`
と照合する。先にprofile、基準binary、基準12条件のWAVとreport、登録入力の全hashを検査し、
一つでも無ければ取得前に止める。

リポジトリのルートで実行する。`OUT`は存在しない新しい名前にする。

```sh
set -euo pipefail
export D=docs/roadmap/temporal-dcc/i11-inputs
export OUT=target/i11-same-record-$(date +%Y%m%d-%H%M%S)
export BASE=target/i11-stage1-54b-20260923/baseline
export PROFILE=/home/shafi/lwrk/conchordal/target/i10-action-profiles-20260918.bin

python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

if not __debug__:
    raise SystemExit('Pythonのassert検査が無効。PYTHONOPTIMIZEを外す')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

plan = json.loads((Path(os.environ['D']) / 'plan.json').read_text())
cases = plan['cases']
assert len(cases) == 12 and len(set(cases)) == 12
assert not Path(os.environ['OUT']).exists(), 'OUT must be new'
profile = Path(os.environ['PROFILE'])
assert profile.is_file() and sha256(profile) == '033905eadd9a7c681a8568ea347c25d1c1482d49d708bc849b00607a85d51fb5', profile
for name in ('body', 'proxy', 'none'):
    config = Path(os.environ['D']) / 'inputs' / f'config-{name}.toml'
    assert str(profile) in config.read_text(), config
for name, expected in plan['inputs_sha256'].items():
    path = Path(name)
    assert path.is_file() and sha256(path) == expected, path
base = Path(os.environ['BASE'])
binary = base / 'render'
assert binary.is_file() and sha256(binary) == plan['binaries']['render']['sha256'], binary
for case in cases:
    for suffix in ('wav', 'jsonl'):
        path = base / f'{case}-none-render-0.{suffix}'
        assert path.is_file() and path.stat().st_size > 0, path
print('登録入力、profile、基準binary・12条件のWAV/reportを確認')
PY

cargo build --release --bin conchordal --bin conchordal-render
mkdir "$OUT"
python3 "$D/acquire.py" --root "$OUT" --plan "$D/plan.json" \
  --mode render --variant body --variant proxy --variant none
```

これで12条件×3設定のoffline renderを各1回、計36実行する。出力先の`binaries.json`と
`source-sha256.json`が実際のbinaryとソースを記録する。次の検査は各判定scriptを走らせた後、
そのJSONの件数と内容を読む。`bit_identity.py`はWAV欠損でも`pass`を出し得るため、
36本のWAV・reportの存在を先に確かめる。`verify_i11_stage1.py`は決定0件のreportで終了コード1を返す。
登録した`sine-hold` 3条件の`body`／`proxy`計6本だけを決定0件として許し、検査項目の失敗は一切許さない。

```sh
python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

if not __debug__:
    raise SystemExit('Pythonのassert検査が無効。PYTHONOPTIMIZEを外す')

root = Path(os.environ['OUT'])
base = Path(os.environ['BASE'])
docs = Path(os.environ['D'])
cases = json.loads((docs / 'plan.json').read_text())['cases']

def run(args, output, allowed=(0,)):
    result = subprocess.run(['python3', *map(str, args), '--output', output],
                            capture_output=True, text=True)
    if result.returncode not in allowed:
        raise SystemExit(result.stdout + result.stderr)
    return json.loads(Path(output).read_text()), result.returncode

ledger = json.loads((root / 'runs.json').read_text())['runs']
expected = {(case, variant, 'render', 0)
            for case in cases for variant in ('body', 'proxy', 'none')}
assert len(ledger) == 36
assert {(r['case'], r['variant'], r['mode'], r['rep']) for r in ledger} == expected
assert all(r['exit'] == 0 for r in ledger)
for case, variant, _, _ in expected:
    for suffix in ('wav', 'jsonl'):
        path = root / f'{case}-{variant}-render-0.{suffix}'
        assert path.is_file() and path.stat().st_size > 0, path

api_path = root / 'api-effect.json'
subprocess.run(['python3', str(docs / 'api_effect.py'), '--root', str(root),
                '--plan', str(docs / 'plan.json'), '--out', str(api_path)], check=True)
api = json.loads(api_path.read_text())
assert api['cases_with_comparison'] == 12
assert {r['case'] for r in api['cases']} == set(cases)
for row in api['cases']:
    a, b = row['a_body_vs_proxy'], row['b_body_vs_none']
    assert 'skipped' not in a and 'skipped' not in b
    assert a['wav'] is not None and b['wav'] is not None
    if row['case'].startswith(('harmonic-', 'modal-')):
        assert a['t1_effect_observed'] and a['selected_offset']['differing'] > 0
        assert a['wav']['differs']
    else:
        assert a['selected_offset']['differing'] == 0 and not a['wav']['differs']

pairs = [path for case in cases for path in
         (root / f'{case}-body-render-0.jsonl', root / f'{case}-proxy-render-0.jsonl')]
locator, _ = run(['scripts/locate_i11_first_divergence.py', *pairs],
                 str(root / 'first-divergence.json'))
assert len(locator) == 12
for case, row in zip(cases, locator):
    assert row['first_prior_input_difference'] is None, case
    if case.startswith(('harmonic-', 'modal-')):
        assert row['status'] == 'diverged' and row['sources_valid'], case
        assert row['powers_differ'] and not row['other_differing_inputs'], case
    else:
        assert row['status'] == 'no_divergence', case
        if case.startswith('sine-flow-'):
            assert row['compared'] > 0, case

bit_path = root / 'bit-identity.json'
subprocess.run(['python3', str(docs / 'bit_identity.py'), '--root', str(root),
                '--plan', str(docs / 'plan.json'), '--baseline-dir', str(base),
                '--out', str(bit_path)], check=True)
bit = json.loads(bit_path.read_text())
assert bit['pass'] and bit['cases_checked'] == 12 and bit['cases_skipped'] == 0
assert {r['case'] for r in bit['cases']} == set(cases)
assert all(r['pass'] and r['wav'] is not None and r['wav']['equal']
           and len(r['types']) == 10 and all(t['equal'] for t in r['types'].values())
           for r in bit['cases'])

candidate_pairs = [path for case in cases for path in
                   (base / f'{case}-none-render-0.jsonl', root / f'{case}-none-render-0.jsonl')]
candidates, _ = run(['scripts/compare_i11_candidate_records.py', *candidate_pairs],
                    str(root / 'candidate-records.json'))
assert len(candidates) == 12
assert all(r['matched'] > 0 and r['differing'] == 0
           and r['identical'] == r['matched'] for r in candidates)

verified, status = run(['scripts/verify_i11_stage1.py', *pairs],
                       str(root / 'verify-stage1.json'), allowed=(1,))
assert status == 1 and len(verified) == 24
assert set(verified) == {str(p) for p in pairs}
zero_allowed = {str(root / f'{case}-{variant}-render-0.jsonl')
                for case in cases if case.startswith('sine-hold-')
                for variant in ('body', 'proxy')}
assert {name for name, row in verified.items() if row['decisions'] == 0} == zero_allowed
assert all(all(check['failed'] == 0 for check in row['checks'].values())
           for row in verified.values())
print('36 render、§5.4(a)(b)、§5.5(a)、24 reportの独立参照を確認')
PY
```

locatorの終了コード0は、12対がそろったことやharmonic／modalで選択差が出たことを保証しない。
上のJSON検査まで通った場合だけ、登録条件での§5.5(a)再判定に使う。§5.4(b)の片側だけの候補recordは
既定の規則どおり件数を報告し、内容一致の分母には含めない。

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
