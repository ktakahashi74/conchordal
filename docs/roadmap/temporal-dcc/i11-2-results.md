# I11 第2段：結果記録 — I11-2限定技術完了（追加証拠による事後受入）、原§5.7は11欄不合格

この記録は、実装前登録の[I11 第2段](i11-2-arrival-comparison.md)と[元登録の§4.5・§5・§6](i11-onset-comparison.md)に対し、取得済みの証拠と2026-09-24の事後受入判断を記す。offline 96本と§5.9のgap分布を取得・集計し、A/A 504本からfloorを固定した。原試行の性能主判定560欄は11欄不合格で、到来使用の適格性36件は全合格。この原§5.7の不合格判定は維持する。別に封印したAC給電・覚醒統制下の追加252本は同じfloorで560欄と36件のすべてに適合した。[独立結果レビュー](../../../target/i11-stage2-awake-diagnostic-20260924/results-review.md)と[独立受入審査](../../../target/i11-stage2-awake-diagnostic-20260924/acceptance-review.md)を経て、後段§6の範囲で**I11-2限定技術完了（追加証拠による事後受入）**を採用した。無限定な原登録試行の合格、作者による生成作用の採用、O15／O18全体の達成は宣言しない。既定値は `None` のまま。

## 取得条件と版

実装前の入力・判定規則は 2026-09-24 03:50:08 UTC に凍結した。旧12条件と新しい外部パルス6条件を、登録した順序で計96本の offline render として取得した。旧12条件は元の `None` 12本、第2段前の `body-off`／`proxy-off` 24本、候補版の `none`／`body-off`／`proxy-off`／`body-on` 48本。新6条件は候補版の `body-off`／`body-on` 12本。旧12条件のseedは20260918、新6条件は20260924。いずれも48 kHz、演奏6秒と終端0.5秒。新条件は sine／harmonic／modal の各4・16 Voice、2 Hz の habitat 専用pulseを含む。

凍結 `plan.json` SHA-256 は `1ba46d0955aedf8656bf222d956ab619d4ecdae2fe6555f96b989abb61fd7c95`。候補の[封印manifest](../../../target/i11-stage2-20260924/candidate/manifest.json) SHA-256 は `e3b76d987ddb4b7a479d5ac3927d2c27d2064af1a40f229a4f51fa971e4d45f4`。候補の基点commitは `7f76e0b970b1ed06ebce1a60e9cee71f35ecbb8f` だが、実装後の作業treeは未commitであり、版の同定にはmanifest内のsource別SHAを使う。候補 `conchordal-render` SHA-256 は `1d1d810f234d13d0947cba507715d2a94aaae11627fef82a457dc48ba2ce4e1e`、`conchordal` は `1d0d306eaa138a74b61a22c02fbd84aeec1ea942af5963973c8b8c08f36c4523`。取得provenance SHA-256 は `e1fdd70fe8bfab8ff4b58e064ce5095272acbc566de04e717cc3f37255099660`、96本のledger SHA-256 は `73b6ef17fe259f94fab53681b27341e7f9c6de3a59a380875c07f0a020bf8d2b`。[offline評価](../../../target/i11-stage2-20260924/offline-evaluation.json) は96本を検査し、失敗0・exit 0。取得中に実行記録が追記される `acquisition/runs.json` の先頭96本分を、後日、封印済み取得器と同じJSON形式で[復元した](../../../target/i11-stage2-20260924/offline-ledger-96-reconstructed.json)。[復元metadata](../../../target/i11-stage2-20260924/offline-ledger-reconstruction.json)は、そのSHA-256が既存のoffline評価に記録された `73b6ef17fe259f94fab53681b27341e7f9c6de3a59a380875c07f0a020bf8d2b` と一致したことを示す。これはA/A開始前に保存した元の96本分のsnapshotではない。

## 実装・数値核の検査

候補版の全 `cargo test` は1176成功・0失敗・37 ignored、exit 0。`cargo clippy -- -D warnings` もexit 0。候補封印前のAstra独立再実行では、第2段関連Python検査35件が成功した。保存済みのI11 Pythonログは66件成功、offline評価器のfocused検査ログは8件成功だが、対象と実行時点が異なるため件数を合算しない。独立CDF照合は1831 checks・失敗0。これは数値核と強制fixtureの証拠であり、実走に現れなかった失敗条件の自然発生を意味しない。根拠は[候補validation](../../../target/i11-stage2-20260924/candidate/validation/)と[独立監査](../../../target/i11-stage2-20260924/completion-audit-pending.md)。 [offline96の独立Astra結果レビュー](../../../target/i11-stage2-20260924/offline-result-review.md)は、登録した§5.4a／b・T1・T2の集計に適合と判定した。生report・WAVの再走査ではなく、保存済み集計と取得前の検査器監査に基づくレビュー。[最終封印監査](../../../target/i11-stage2-20260924/final-seal-audit.json)は候補manifest掲載ファイル490／490、binary 2件、validation 12件、build log 1件、取得provenanceと、取得が参照した入力24／24件、floor／snapshotのSHA整合を確認し、性能の合格判定とは分けている。

## §5.4a／§5.4b：基準の保存

元の `None` と候補 `None` の旧12条件は、WAV SHAと登録した10種の学習recordが全条件で一致した。比較から除いたのは登録済みの実時間欄、および `participation_context` の新8欄だけ。第2段前と候補の `body-off`／`proxy-off` 旧24条件も、WAVと決定・学習recordが全条件で一致した。飽和に伴う一方のみのcandidateは別集計とし、不変比較へ混ぜていない。

共通candidate recordの比較は36組で34,610件一致し、差分0件（bit差分0件）。基準側のみ532件、候補側のみ608件。凍結packetの再評価は実施していない。改訂登録§5.4bは共通record比較を採用しており、この未実施は限界として残す。

## §5.5：同一状態介入の作用

T1の旧12条件では、必須のharmonic／modal各4・16・64 Voiceの6条件すべてで、最初の分岐に至る入力の違いはbodyとproxyのpowerだけだった。最初に異なった出力は `selected_offset`、WAVも全6条件で異なる。[補助確認](../../../target/i11-stage2-20260924/t1-selected-offset-check.json)と[独立結果レビュー](../../../target/i11-stage2-20260924/offline-result-review.md)でも選択差を確認した。indexは決定列の0始まり。sineのflow 3条件は決定があるが選択・WAV差はなく、hold 3条件は決定0件でWAV差もない。これらも評価器に報告済みであり、必須6条件の作用に含めない。

| T1条件 | 最初の分岐index | 異なる出力 | 他の入力差 | WAV差 |
|---|---:|---|---:|---|
| harmonic-flow-4 | 10 | `selected_offset` | 0 | あり |
| harmonic-flow-16 | 10 | `selected_offset` | 0 | あり |
| harmonic-flow-64 | 31 | `selected_offset` | 0 | あり |
| modal-flow-4 | 5 | `selected_offset` | 0 | あり |
| modal-flow-16 | 10 | `selected_offset` | 0 | あり |
| modal-flow-64 | 31 | `selected_offset` | 0 | あり |

T2の新6条件では、`body-off`／`body-on` の非到来入力prefixがbit一致し、`body-on` の同じCDFを使う独立介入検査に失敗はなかった。既知CDFを使った決定は各条件で確認した。各身体について少なくとも1条件で選択とWAVが変わった。分岐なしの2条件も表に残す。

| T2条件 | 既知到来使用決定数 | 最初の分岐index | 出力差 | WAV差 |
|---|---:|---:|---|---|
| sine-pulse-flow-4 | 23 | なし | なし | なし |
| sine-pulse-flow-16 | 98 | 86 | `selected_offset` | あり |
| harmonic-pulse-flow-4 | 23 | なし | なし | なし |
| harmonic-pulse-flow-16 | 126 | 130 | `selected_offset` | あり |
| modal-pulse-flow-4 | 23 | 26 | `selected_offset` | あり |
| modal-pulse-flow-16 | 125 | 85 | `selected_offset` | あり |

## 既知の限界

到来係数は登録した診断値であり、音楽的にfit済みとは扱わない。CDFは実観測に由来する群到来の周辺量であり、実kick・実seed・実rhythmsに条件付けたI10予測の消費ではない。自己除外は識別・期限が一致するBindingに限定される。旧sine-holdの3条件はdecisionが0件で、作用の根拠には含めない。§5.4bの凍結packet再評価は未実施。今回のoffline結果から聴取による採用、資源受入、liveの行動効果を推論しない。

## §5.9：実発行ToneEnergyとのgap分布（取得・集計済み）

[gap-summary.json](../../../target/i11-stage2-20260924/gap-summary.json)（SHA-256 `3fc7b1fe458e818ee0b10319b058c5a958ef6e1b2c015d0560d26afcab26c9bd`）は旧12・新6条件の18件を含む。取得は[status](../../../target/i11-stage2-20260924/gap-acquisition.status.txt)でexit 0、[log](../../../target/i11-stage2-20260924/gap-acquisition.log)で対象Rust test 1件成功。実行したdebug test harnessのSHA-256は `942b847725998ec545735cd1db9edc4a96dfb3ad7a85d300daba4ed1ea0e013b`。候補manifestの封印source 215ファイルとの照合は不一致0件。発行された実ToneEnergyを同じ機会のDrepへ再投影した診断値で、`power_gap`、`overlap_gap`、`term_gap` の順に中央値／p95／最大を示す。`n` はその行の `power_gap` 件数。[4母集団のpool値](../../../target/i11-stage2-20260924/gap-populations.json)（SHA-256 `6e477222402f8451e8525ecea48b90fff51375cf8b068c8b9c7888dd66d137d5`）は全18条件のcase別出力と一致した。[取得・集計provenance](../../../target/i11-stage2-20260924/gap-provenance.json)（SHA-256 `c17aab429714b2cfb4909c7c2fab9622d1cef52de74c40391116f37b65b65673`）に実行command、script SHA、各artifact SHAを保存した。表では旧条件と新条件、`body` とproxyを別母集団とする。全精度値とcase別の除外理由はJSONに保持する。

| 母集団 | power gap n・中央／p95／最大 | overlap gap n・中央／p95／最大 | term gap n・中央／p95／最大 |
|---|---:|---:|---:|
| 旧12 body | 3,343: 0.0272078775 / 0.0957023528 / 0.1338848446 | 3,343: 0.0000533981 / 0.0004948400 / 0.0032631911 | 3,343: 0.0001264957 / 0.0017635201 / 0.0128509748 |
| 旧12 proxy | 48: 0.7773778283 / 0.9999999679 / 0.9999999681 | 9: 0 / 0 / 0 | 9: 0 / 0 / 0 |
| 新6 body | 686: 0.0285233898 / 0.0968506087 / 0.1310051551 | 686: 0.0000000339 / 0.0000020581 / 0.0007147086 | 686: 0.0000159848 / 0.0005303125 / 0.0025210860 |
| 新6 proxy | 9: 0.6529866738 / 0.9999999687 / 0.9999999687 | 6: 0 / 0 / 0 | 6: 0 / 0 / 0 |

proxyのpower母集団は `proxy(absent)` と `proxy(stale)` の合計。重なりと費用項はstaleのみ（旧9件、新6件）で、absent（旧39件、新3件）へは適用していない。これらのpool分位値はcase中央値の平均ではなく、保存した個別gap値から求めた値。gap再投影は実routeをそのまま使う処理とは別で、登録どおりbus0用の `[true, false]` を強制した。`Drep.ceil` による整数区間への射影を使い、連続時間上のDrep端点を厳密再現したとは主張しない。[全Dの整数境界監査](../../../target/i11-stage2-20260924/gap-boundary-audit.json)は18条件の投影4,086件について各17境界、計69,462比較を実施し、不一致0件・exit 0だった。3種類のDについて `ceil(Fraction.from_float(D) * k / 16)` と `ceil(ceil(D) * k / 16)` が一致した。この実入力での整数境界の同値であり、任意のDや連続時間端点の同値ではない。[gapの独立Astra結果レビュー](../../../target/i11-stage2-20260924/gap-result-review.md)が残した境界確認をこの監査で補った。

### 旧12条件

| 条件 | footprint出所 | n | power gap 中央／p95／最大 | overlap gap 中央／p95／最大 | term gap 中央／p95／最大 |
|---|---|---:|---:|---:|---:|
| harmonic-flow-16 | body | 204 | 0.0869816 / 0.104591 / 0.130505 | 0.000483556 / 0.00100578 / 0.00134644 | 0.000786892 / 0.00694701 / 0.012851 |
| harmonic-flow-16 | proxy(absent) | 1 | 0.779245 / 0.779245 / 0.779245 | — | — |
| harmonic-flow-16 | proxy(stale) | 1 | 0.67579 / 0.67579 / 0.67579 | 0 / 0 / 0 | 0 / 0 / 0 |
| harmonic-flow-4 | body | 56 | 0.0862158 / 0.0977238 / 0.100489 | 0.00135741 / 0.00302805 / 0.00326319 | 0.00425539 / 0.00923513 / 0.0108152 |
| harmonic-flow-4 | proxy(absent) | 1 | 0.779245 / 0.779245 / 0.779245 | — | — |
| harmonic-flow-64 | body | 856 | 0.0870218 / 0.106863 / 0.133885 | 0.000134113 / 0.000312711 / 0.000580692 | 0.000204715 / 0.00249342 / 0.0115817 |
| harmonic-flow-64 | proxy(absent) | 11 | 0.778318 / 0.779726 / 0.779726 | — | — |
| harmonic-flow-64 | proxy(stale) | 2 | 0.65082 / 0.67579 / 0.67579 | 0 / 0 / 0 | 0 / 0 / 0 |
| modal-flow-16 | body | 202 | 0.0023777 / 0.00898315 / 0.0153591 | 2.4968e-07 / 9.96458e-07 / 1.55417e-06 | 2.94374e-05 / 0.000129516 / 0.000247197 |
| modal-flow-16 | proxy(absent) | 1 | 1 / 1 / 1 | — | — |
| modal-flow-16 | proxy(stale) | 1 | 0.996991 / 0.996991 / 0.996991 | 0 / 0 / 0 | 0 / 0 / 0 |
| modal-flow-4 | body | 56 | 0.00202245 / 0.00826341 / 0.0103455 | 4.88211e-07 / 2.0165e-06 / 2.50923e-06 | 1.99626e-05 / 0.000106374 / 0.000200363 |
| modal-flow-4 | proxy(absent) | 1 | 1 / 1 / 1 | — | — |
| modal-flow-64 | body | 854 | 0.0021824 / 0.0106262 / 0.0224057 | 6.05935e-08 / 3.30592e-07 / 6.20771e-07 | 1.04923e-05 / 6.83437e-05 / 0.000516682 |
| modal-flow-64 | proxy(absent) | 11 | 1 / 1 / 1 | — | — |
| modal-flow-64 | proxy(stale) | 2 | 0.996991 / 0.997314 / 0.997314 | 0 / 0 / 0 | 0 / 0 / 0 |
| sine-flow-16 | body | 204 | 0.0269074 / 0.0387171 / 0.0430135 | 0.000169042 / 0.000420798 / 0.000579888 | 0.000474052 / 0.00130726 / 0.00152408 |
| sine-flow-16 | proxy(absent) | 1 | 0.108943 / 0.108943 / 0.108943 | — | — |
| sine-flow-16 | proxy(stale) | 1 | 0.190474 / 0.190474 / 0.190474 | 0 / 0 / 0 | 0 / 0 / 0 |
| sine-flow-4 | body | 56 | 0.0264418 / 0.0367124 / 0.0395985 | 0.000395489 / 0.00077252 / 0.000887482 | 0.000823238 / 0.00138538 / 0.00150582 |
| sine-flow-4 | proxy(absent) | 1 | 0.108943 / 0.108943 / 0.108943 | — | — |
| sine-flow-64 | body | 855 | 0.0272875 / 0.0422726 / 0.0525949 | 5.2304e-05 / 0.000158574 / 0.000323759 | 0.000160606 / 0.000803253 / 0.00166411 |
| sine-flow-64 | proxy(absent) | 11 | 0.108943 / 0.112219 / 0.112219 | — | — |
| sine-flow-64 | proxy(stale) | 2 | 0.190474 / 0.195043 / 0.195043 | 0 / 0 / 0 | 0 / 0 / 0 |
| sine-hold-16 | 該当なし | 0 | — | — | — |
| sine-hold-4 | 該当なし | 0 | — | — | — |
| sine-hold-64 | 該当なし | 0 | — | — | — |

旧12条件では投影3,391件（body 3,343、proxy(absent) 39、proxy(stale) 9）。sine-hold 4／16／64の `no_decision` は計84件で、表のgap母集団から除外。`proxy(absent)` 39件は `power_gap` のみ算出し、`overlap_gap` と `term_gap` は適用していない。

### 新6条件

| 条件 | footprint出所 | n | power gap 中央／p95／最大 | overlap gap 中央／p95／最大 | term gap 中央／p95／最大 |
|---|---|---:|---:|---:|---:|
| harmonic-pulse-flow-16 | body | 195 | 0.0866728 / 0.111909 / 0.131005 | 3.93186e-07 / 7.92574e-06 / 0.000714709 | 0.000172919 / 0.000854107 / 0.00252109 |
| harmonic-pulse-flow-16 | proxy(absent) | 1 | 0.777636 / 0.777636 / 0.777636 | — | — |
| harmonic-pulse-flow-16 | proxy(stale) | 1 | 0.652987 / 0.652987 / 0.652987 | 0 / 0 / 0 | 0 / 0 / 0 |
| harmonic-pulse-flow-4 | body | 34 | 0.0886273 / 0.120713 / 0.125269 | 4.02609e-07 / 1.75934e-06 / 2.95332e-06 | 0.000284531 / 0.00101981 / 0.00113066 |
| harmonic-pulse-flow-4 | proxy(stale) | 1 | 0.652987 / 0.652987 / 0.652987 | 0 / 0 / 0 | 0 / 0 / 0 |
| modal-pulse-flow-16 | body | 194 | 0.00266805 / 0.0145639 / 0.0219488 | 1.73778e-08 / 1.57091e-07 / 5.57156e-07 | 7.37845e-06 / 3.96999e-05 / 0.000148621 |
| modal-pulse-flow-16 | proxy(absent) | 1 | 1 / 1 / 1 | — | — |
| modal-pulse-flow-16 | proxy(stale) | 1 | 0.997332 / 0.997332 / 0.997332 | 0 / 0 / 0 | 0 / 0 / 0 |
| modal-pulse-flow-4 | body | 34 | 0.00158845 / 0.0110673 / 0.0119714 | 7.33885e-09 / 2.02793e-07 / 5.67652e-07 | 5.5768e-06 / 7.86962e-05 / 0.000158476 |
| modal-pulse-flow-4 | proxy(stale) | 1 | 0.997332 / 0.997332 / 0.997332 | 0 / 0 / 0 | 0 / 0 / 0 |
| sine-pulse-flow-16 | body | 195 | 0.0284683 / 0.042442 / 0.0490217 | 1.42793e-08 / 6.03544e-07 / 0.000139201 | 9.29105e-06 / 5.71222e-05 / 0.000774968 |
| sine-pulse-flow-16 | proxy(absent) | 1 | 0.114952 / 0.114952 / 0.114952 | — | — |
| sine-pulse-flow-16 | proxy(stale) | 1 | 0.207824 / 0.207824 / 0.207824 | 0 / 0 / 0 | 0 / 0 / 0 |
| sine-pulse-flow-4 | body | 34 | 0.0285267 / 0.0430685 / 0.0430723 | 2.21193e-08 / 1.12957e-07 / 1.97428e-07 | 1.40289e-05 / 5.30453e-05 / 7.59179e-05 |
| sine-pulse-flow-4 | proxy(stale) | 1 | 0.207824 / 0.207824 / 0.207824 | 0 / 0 / 0 | 0 / 0 / 0 |

新6条件では投影695件（body 686、proxy(absent) 3、proxy(stale) 6）。発行onsetに対応する非skipの参加決定が見つからなかった `no_decision` は計49件で、表のgap母集団から除外。`proxy(absent)` 3件も `power_gap` のみ算出し、重なりと費用項は適用していない。gapの大小に合格閾値や身体の優越性を設定しない。

## §5.7：固定floorによる性能主判定は11欄不合格

A/Aは504本ともexit 0。3 passそれぞれ168本で、各passの28条件（case×report有無）につき `none-A`／`none-B` の各3反復、計6本を取得した。[A/A floor](../../../target/i11-stage2-20260924/acquisition/aa-floor.json)のSHA-256は `ac737802286730f020cb62dbd01220397f484883adcf66c149edbc599990c00f`。各腕の3反復を処理時間では中央値、超過hop数では合計にまとめ、84条件差（28条件×3 pass）の絶対値から欄ごとの最大を登録どおりfloorに採用した。最終A/A run 600の終了は2026-09-24 06:31:57.063628 UTC、floorの固定は同日06:31:58.744258 UTC。[取得status](../../../target/i11-stage2-20260924/aa-acquisition.status.txt)はexit 0。[600本時点のledger snapshot](../../../target/i11-stage2-20260924/acquisition-ledger-through600.json)のSHA-256は `2fb4611e0b0a3cb97c50bfb72162674a2a8b3e990fb31ea63bfc1e94367e4821`。性能比較の取得前にfloorを固定した。

| 欄 | floor | pass 3でpass 1・2の全28条件最大を超えた条件数／28 |
|---|---:|---:|
| `population_us_median` | 82.333 µs | 3 |
| `population_us_p99` | 1,736.542 µs | 2 |
| `population_us_max` | 2,814.958 µs | 0 |
| `synthesis_us_median` | 1,445.125 µs | 4 |
| `synthesis_us_p99` | 5,589.291 µs | 1 |
| `synthesis_us_max` | 54,149.458 µs | 0 |
| `elapsed_us_median` | 2,517.083 µs | 4 |
| `elapsed_us_p99` | 10,288.083 µs | 0 |
| `elapsed_us_max` | 58,280.167 µs | 0 |
| `over_budget_hops` | 318 hop | 1 |

右列はrun数ではなく、各欄で先行2 passの全28条件から作った最大値を第3 passが超えた条件数。10欄の延べ超過は15／280欄×条件、少なくとも1欄で超過した異なる条件は4／28。したがって第3 passもfloorを広げている。特に `elapsed_us_max` のfloorは58.28 msと大きい。このfloorは登録した工学的判定幅であり、統計的同等性や実機リアルタイム性能の合格を意味しない。結果に合わせた変更は行わない。

性能比較252本を含む全852本の実行はexit 0で、最終runの終了は2026-09-24 12:45:34.652132 UTCだった。[852本のledger snapshot](../../../target/i11-stage2-20260924/acquisition-ledger-through852.json)のSHA-256は `5819c1fdbba7a726ab8ab2dc5fec78332da5e0a97eb4b479b92023d366305a29`。600本時点のprefixと固定floorのSHAは変わっていない。

[封印済み性能検査器の結果](../../../target/i11-stage2-20260924/performance-verification.json)は596 checks中11件失敗、[終了status](../../../target/i11-stage2-20260924/performance-verification.status.txt)はexit 1。内訳は主判定560欄中549合格・11不合格、到来使用の適格性36件は全合格。主判定は28条件中4条件、56比較中7比較で少なくとも1欄が不合格だった。report無し側の280欄は全合格、report有り側で11欄が不合格。元のfloor無し規則では560欄中324合格・236不合格（対None 122欄、対body-off 114欄が不合格）。登録どおり絶対差を判定するため、on側が小さい差も許容超過なら不合格となる。

以下は不合格の全11欄。すべてreport有り。時間の単位はµs。on・基準は各runの全hop統計を求めた後の3反復間中央値であり、全runをpoolした最大値ではない。許容は登録の5%または1,000µsと固定floorから定まり、結果に応じて変更していない。

| 条件 | 基準 | 欄 | on | 基準値 | 絶対差 | 許容 |
|---|---|---|---:|---:|---:|---:|
| sine-hold-16 | body-off | `population_us_max` | 9,278.709 | 4,018.958 | 5,259.751 | 2,814.958 |
| sine-hold-16 | body-off | `synthesis_us_max` | 137,417.125 | 3,530.125 | 133,887.000 | 54,149.458 |
| sine-hold-16 | body-off | `elapsed_us_max` | 145,226.167 | 8,135.000 | 137,091.167 | 58,280.167 |
| sine-flow-4 | none | `elapsed_us_max` | 136,721.750 | 13,409.125 | 123,312.625 | 58,280.167 |
| sine-flow-4 | body-off | `elapsed_us_max` | 136,721.750 | 13,456.875 | 123,264.875 | 58,280.167 |
| sine-flow-16 | none | `synthesis_us_max` | 130,739.750 | 13,824.583 | 116,915.167 | 54,149.458 |
| sine-flow-16 | body-off | `synthesis_us_median` | 2,651.750 | 4,581.291 | 1,929.541 | 1,445.125 |
| sine-flow-16 | body-off | `synthesis_us_max` | 130,739.750 | 16,792.750 | 113,947.000 | 54,149.458 |
| sine-flow-16 | body-off | `elapsed_us_max` | 134,039.417 | 24,725.916 | 109,313.501 | 58,280.167 |
| harmonic-flow-4 | none | `elapsed_us_max` | 133,958.708 | 16,819.375 | 117,139.333 | 58,280.167 |
| harmonic-flow-4 | body-off | `elapsed_us_max` | 133,958.708 | 17,171.042 | 116,787.666 | 58,280.167 |

### 取得後の資源・遅延診断

[診断集計](../../../target/i11-stage2-20260924/diagnostic-summary.json)（SHA-256 `92aae56391d8bf5c48ea3208467f88d5c71dd4047b2cda00859ea5c990ad76fc`）は[実行status](../../../target/i11-stage2-20260924/diagnostic-summary.status.txt)がexit 0で、852本のledgerとartifactを照合し、性能252本・84条件を収録した。各runのcase、設定、report有無、反復、両busの資源欄と候補workerの全counterはJSONに残す。以下の合計と範囲は記述値であり、登録済みの性能合否規則を変更しない。旧8条件と新6条件、report有無、`none`／`body-off`／`body-on`を区別する。

到来表は`body-on`のbus 0だけで生成された。`none`／`body-off`とbus 1の`arrival_resources`は`null`。profile全体の到来既知決定数も`none`／`body-off`では元profileの欄が省略され、診断JSONでは`null`となる。これを0へ置き換えない。下表は各行24本（旧8条件）または18本（新6条件）の合計。`cdf_failed`は群のCDFが`None`になった回数で、`no_forecast`や非周期などを含み、数値積分の失敗件数とは同義でない。`generated`と`failed`は群単位なのでpublication数より多くなり得る。`horizon_overflow`は各行0。`generation_us_last`を含むrun別の全欄はJSONにある。

| 母集団・mode（body-on／bus 0） | runs | publication | CDF有／無 | 評価回数 | 生成合計／単発最大 | 既知決定 |
|---|---:|---:|---:|---:|---:|---:|
| 旧8・report | 24 | 14,664 | 9,042／46,932 | 20,609,184 | 972.226／115.608 ms | 264 |
| 旧8・no-report | 24 | 14,664 | 9,042／46,932 | 20,609,184 | 810.711／0.846 ms | 264 |
| 新6・report | 18 | 10,998 | 14,112／9,630 | 19,584,552 | 985.081／0.472 ms | 1,254 |
| 新6・no-report | 18 | 10,998 | 14,112／9,630 | 19,584,552 | 1,003.324／0.353 ms | 1,254 |

新6の`body-on`はreport有無それぞれ18本で、**各run**の既知到来使用が23〜126件。旧8には発音時刻決定のないsine-holdがあり、既知件数0のrunもある。下表は`body-on`の各runを旧／新とmode別に分け、両busのobserver資源を記す。全252本で各busは611 frameを受領・処理し、source欠測sample、配送drop frame、rejected frameはいずれも各runで0。`frames`と`delivery`の合計時間は別欄で、CDF生成時間を加算し直していない。表の時間範囲はrunごとのtotalの最小〜最大、単発最大はその母集団のrun最大。17 bucketのhistogramと上端（100 µs〜1 s）はJSONに保持し、bucketから精密な中央値・p99を作っていない。

| 母集団・mode（body-on） | bus | frame total／run | frame単発最大 | delivery total／run | delivery単発最大 |
|---|---:|---:|---:|---:|---:|
| 旧8・report | 0 | 82.05〜653.27 ms | 116.01 ms | 84.88〜657.53 ms | 116.03 ms |
| 旧8・report | 1 | 81.22〜459.72 ms | 14.91 ms | 84.28〜471.26 ms | 14.91 ms |
| 旧8・no-report | 0 | 80.57〜544.97 ms | 14.78 ms | 82.59〜549.09 ms | 14.78 ms |
| 旧8・no-report | 1 | 61.01〜458.70 ms | 14.79 ms | 63.26〜462.93 ms | 14.80 ms |
| 新6・report | 0 | 156.53〜302.97 ms | 3.12 ms | 159.02〜308.21 ms | 3.14 ms |
| 新6・report | 1 | 66.42〜484.38 ms | 12.47 ms | 69.12〜500.02 ms | 12.48 ms |
| 新6・no-report | 0 | 161.86〜269.82 ms | 2.49 ms | 164.49〜275.32 ms | 2.50 ms |
| 新6・no-report | 1 | 68.43〜494.72 ms | 10.48 ms | 71.08〜500.74 ms | 10.49 ms |

hopのanalysis／listener待ちはprofileのraw 611 hopから各runで中央値・p99・最大を算出した。次の各「中央値範囲」「p99範囲」はrun別値の最小〜最大、「最大」はrun別最大値の最大であり、複数runをpoolした分位値ではない。比較設定も含む252本すべての値は診断JSONにある。

| 母集団・mode（body-on） | analysis待ち：中央値範囲／p99範囲／最大 | listener待ち：中央値範囲／p99範囲／最大 |
|---|---:|---:|
| 旧8・report | 920.08〜1,843.79／1,569.96〜11,894.79／161,440.46 µs | 5.58〜10.79／6.62〜3,445.46／19,650.96 µs |
| 旧8・no-report | 916.79〜1,864.46／1,543.46〜8,887.12／16,205.71 µs | 4.88〜12.75／7.21〜20.00／3,069.33 µs |
| 新6・report | 1,024.25〜1,600.08／2,032.38〜4,170.42／12,176.00 µs | 6.00〜266.12／8.38〜6,673.50／12,330.92 µs |
| 新6・no-report | 1,025.29〜1,541.46／2,037.21〜3,624.04／14,108.71 µs | 6.12〜267.12／259.71〜6,672.54／8,811.12 µs |

候補workerのcounterは各行のrun合計を示す。`capacity_dropped`はcandidate packetの飽和であり、footprint返却dropとは別。`footprint_requested`と`footprint_completed`は各行で同数だった。`none`のfootprint要求は全runで0。下表にない`unsupported`、`worker_unsupported`、`output_dropped`、`footprint_dropped`、`footprint_output_dropped`、`footprint_resent`、`footprint_released`、`worker_failed`は252本の各runで0またはfalse。`submitted`、`max_processing_us`など全欄のrun別値はJSONにある。

| 母集団 | mode | 設定 | runs | candidate提出／容量drop | footprint要求／完了 | 置換済み |
|---|---|---|---:|---:|---:|---:|
| 旧8 | report | body-off | 24 | 9,644／16,709 | 360／360 | 23 |
| 旧8 | report | body-on | 24 | 9,683／16,687 | 360／360 | 27 |
| 旧8 | no-report | body-off | 24 | 9,567／16,795 | 360／360 | 26 |
| 旧8 | no-report | body-on | 24 | 9,563／16,819 | 360／360 | 27 |
| 新6 | report | body-off | 18 | 4,962／13,407 | 324／324 | 21 |
| 新6 | report | body-on | 18 | 5,006／13,364 | 324／324 | 21 |
| 新6 | no-report | body-off | 18 | 4,804／13,570 | 324／324 | 24 |
| 新6 | no-report | body-on | 18 | 4,871／13,496 | 324／324 | 23 |

footprint返却遅延はreport有りのflow条件で、封印済み`hop_path.footprint_delay`を使用した。各case×設定の3反復をまとめた遅延分布を計算し、以下はその**case別中央値・p99の範囲**とcase別最大値の最大。したがって下表の範囲は旧／新全caseのpool分位値ではない。時計は48 kHzのsample差をmsへ変換し、observerのwall処理時間とは区別する。`proxy(absent)`の分母は遅延が測れた件数ではなく、同じreport群の`participation_context`件数。遅延計測件数には`body`と`proxy(stale)`を含み、出所別の遅延分位値を示す表ではない。

| 母集団・設定（report／flow） | 遅延計測件数 | case別中央値範囲 | case別p99範囲／最大 | proxy(absent)／context |
|---|---:|---:|---:|---:|
| 旧6 flow・body-off | 2,346 | 0〜64.00 ms | 0〜85.33／85.33 ms | 0／2,346 |
| 旧6 flow・body-on | 2,343 | 0〜64.00 ms | 0〜85.33／85.33 ms | 0／2,343 |
| 新6 flow・body-off | 2,073 | 0〜53.33 ms | 0〜74.67／74.67 ms | 3／2,076 |
| 新6 flow・body-on | 2,073 | 0〜53.33 ms | 0〜74.67／74.67 ms | 3／2,076 |

I6の1 horizon `forecast.probability` と新CDFの支持状態を、`body-on`／reportの群slotで照合した。`決定中群`は同じpublicationが複数決定に現れるたびに数える。`publication群`は**各report内**で同一publication payloadを一度だけ数えた群slotで、別run間の同一payloadは合算時に重複排除していない。両者は異なる分母。I6が`None`でもCDFが存在する群があり、旧8で315／279、新6で1,353／1,197（順に決定中群／publication群）。I6の`None`と新CDFの未知を同一視しない。

| 母集団（body-on／report） | I6 probability | CDF | 決定中群 | publication群 |
|---|---|---|---:|---:|
| 旧8 | None | None | 10,056 | 8,799 |
| 旧8 | Some | Some | 1,548 | 1,389 |
| 旧8 | Some | None | 54 | 36 |
| 旧8 | None | Some | 315 | 279 |
| 新6 | None | None | 1,983 | 1,731 |
| 新6 | Some | Some | 1,221 | 1,065 |
| 新6 | Some | None | 0 | 0 |
| 新6 | None | Some | 1,353 | 1,197 |

採否・除外のreport内訳も`body-on`／reportの旧24本・新18本を分ける。`input_status`と`self_status`は決定単位、`group_status`は決定中の群slot単位、候補Pは候補単位であり、列間の件数を足し合わせない。`known P=0`は両母集団とも候補0件だった。既知0を未知へ置き換えたという意味ではない。

| 母集団 | 決定／既知決定 | input: ready／no_publication | self: no_binding／unrouted／no_profile_key | group: used／nonperiodic／reset_unknown | 候補P: 既知／未知 |
|---|---:|---:|---:|---:|---:|
| 旧8・report | 2,520／264 | 2,502／18 | 2,367／129／24 | 315／11,634／24 | 6,072／51,852 |
| 新6・report | 2,229／1,254 | 2,220／9 | 36／2,172／21 | 1,353／3,204／0 | 28,842／22,407 |

CDF未知理由の群slot出現は旧8で`no_forecast` 10,056件・`reset_unknown` 54件、新6で`no_forecast` 1,983件（いずれも決定中群。publication群では順に8,799・36・1,731件）。ここにはCDFが存在するI6 `None`群を混ぜない。代表のsource／available／issue／receipt sample時計と群世代、その他の採否理由は診断JSONに保存した。
ここにない未来・期限・世代不一致などの強制fixture成功を、自然実走でその理由が発生した件数として扱わない。

旧12条件の候補版`None`に対するWAV SHAの参考比較は次のとおり。`body-off`、`body-on`を別に数え、新6には`None`のoffline基準がないので含めない。差は分割数と打切りの違いも含む。生成された音の相違を示すが、作用の単独原因や作者による採用を証明しない。

| 旧条件 | body-off対None：差あり／条件 | body-on対None：差あり／条件 |
|---|---:|---:|
| sine-hold 4／16／64 | 0／3 | 0／3 |
| sine-flow 4／16／64 | 3／3 | 3／3 |
| harmonic-flow 4／16／64 | 3／3 | 3／3 |
| modal-flow 4／16／64 | 3／3 | 3／3 |

測定中の[OS電源ログ](../../../target/i11-stage2-20260924/power-sleep-wake-events.txt)にはSleep／DarkWakeの反復があり、壁時計上の取得中断と対応する。[観測記録](../../../target/i11-stage2-20260924/power-sleep-observation.json)は影響未確定のまま保存した。[Astraの時計・方法レビュー](../../../target/i11-stage2-20260924/sleep-method-review.md)は、ローカル実装でこのMacのRust `Instant`／Python `monotonic` がsleep中に進行しないことを確認した。登録の総 `elapsed_us` はhop待ちを含むが、sleepを含む壁時計経過時間ではない。rootは2026-09-24 12:29:45 UTCごろ、一時的な `caffeinate -i -t 1800` を開始し、[assertion記録](../../../target/i11-stage2-20260924/power-assertions-after-caffeinate.txt)を保存した。12:30:42 UTCにはrun 714の完了を確認した。この抑止は取得途中の環境介入である。温度・cache・schedulerを含む影響量、およびsleepがrun内かrun間かの区別は未確定。runの選別・再取得や固定floorの変更は行わない。後続の判定と最終宣言は登録したhop時計と固定floorへの適合に限り、無中断liveのdeadline適合を含めない。[完了区間とsleepの対応](../../../target/i11-stage2-20260924/sleep-completion-window-overlaps.json)では、35区間にsleepとの重なりがある。すべて性能比較中であり、前run完了から当該run完了までの区間なのでrun本体への帰属は確定しない。852本終了後にcaffeinateを停止し、12:46:20 UTCに終了を確認した。sleepの説明で今回の不合格を合格へ変更しない。

### 追加診断の取得前固定と結果

性能失敗の原因切り分けとして、[別計画](../../../target/i11-stage2-awake-diagnostic-20260924/plan.json)と[封印記録](../../../target/i11-stage2-awake-diagnostic-20260924/seal.json)を用意した。seal SHA-256は `70054d8277e678588e7d46f239e753fd8f412ab93dfac0ff841b78cc21ccf6a4`。元の252本と同じbinary・入力・順序を使い、出力先だけを別にする。旧floorを固定したまま最大1 blockだけ取得し、途中再開や結果を見た再試行は行わない。新floorやA/Aは追加しない。原252本への事前適用では、全596 checksが封印済み検査器の結果と完全一致した。

追加試行は開始からAC給電・蓋開放・一時idle sleep抑止を要求し、各run前後の電源状態とOSログを保存する。睡眠や電源条件の違反があれば統制不成立とし、原データを保持する。[独立Astraの事前レビュー](../../../target/i11-stage2-awake-diagnostic-20260924/independent-prereview.md)後に取得し、[runner status](../../../target/i11-stage2-awake-diagnostic-20260924/attempt-1/status.json)による開始は2026-09-24 13:08:57.174604 UTC、終了は13:39:28.993064 UTC。252本・artifact 630件が揃い、[追加評価](../../../target/i11-stage2-awake-diagnostic-20260924/attempt-evaluation.json)はexit 0、SHA-256は `6e3c9531d8c276eb18e0dfb958d7c17d2b4f16cdff9c8116fb3de129b4de7244`。[電源状態の生記録](../../../target/i11-stage2-awake-diagnostic-20260924/attempt-1/power-states.jsonl)は開始・終了と各run前後の計506件すべてでAC給電・蓋開放を示し、統制違反のOS電源イベントは0件。追加評価器はrunner status・ledger・OSログのSHA整合を確認し、統制成立と596 checks全合格を別々に記録した。

| 比較 | 登録主判定560欄 | 到来適格性36件 | floorなしの元許容規則560欄 | 統制 |
|---|---:|---:|---:|---|
| 原試行 | 549合格・11不合格 | 36合格 | 324合格・236不合格 | 途中にバッテリー動作・sleep・抑止介入 |
| 追加診断 | 560合格・0不合格 | 36合格 | 321合格・239不合格 | AC給電・蓋開放506／506、違反イベント0 |

追加診断は原試行と同じbinary・入力・run順序・[固定floor](../../../target/i11-stage2-20260924/acquisition/aa-floor.json)を使った。[追加ledger](../../../target/i11-stage2-awake-diagnostic-20260924/attempt-1/runs.json)のSHA-256は `71686909a11774a88bf08addab73643c9f73769bd80c08bf16f3fd708ac74a6d`。元の[主ledger](../../../target/i11-stage2-20260924/acquisition-ledger-through852.json)（SHA-256 `5819c1fdbba7a726ab8ab2dc5fec78332da5e0a97eb4b479b92023d366305a29`）、[主判定](../../../target/i11-stage2-20260924/performance-verification.json)（SHA-256 `f087e0068f6cb32adaa9117af8292d88d93893741af99a8795d6b67d1331f03c`）、固定floor（SHA-256 `ac737802286730f020cb62dbd01220397f484883adcf66c149edbc599990c00f`）は変更していない。原試行と追加診断の環境差はAC給電・覚醒維持を含む複合条件であり、sleep単独の因果効果は識別できない。floor、とくに `elapsed_us_max` の58.28 msは大きな工学的判定幅で、全欄合格から統計的同等性や無中断liveのdeadline適合は推論しない。原試行の11欄不合格と、原試行のみでは§6の技術完了未達という判断は維持する。[追加診断の独立結果レビュー](../../../target/i11-stage2-awake-diagnostic-20260924/results-review.md)を経て、後段で別の事後受入判断を記録する。

## §6：原試行の未達判定と事後限定受入

以下は原試行だけを根拠とした時点の未達判定であり、後続の追加証拠で原試行の合否を書き換えない。

§5.7は全560欄合格を必須としているが、今回の取得では11欄が不合格だった。[独立Astraの性能失敗レビュー](../../../target/i11-stage2-20260924/performance-failure-review.md)は全252 profileから主判定560欄を再計算し、同じ11欄の不合格と到来適格性36／36を確認した。他の検査の成功やsleepの観測でこの要件を代替しない。第2段の技術完了は宣言しない。失敗の原因と追加検証の扱いは別に検討する。作者のA1／A2、R2の全資源受入、O15／O18は別の判断として残す。

2026-09-24、追加252本の[独立結果レビュー](../../../target/i11-stage2-awake-diagnostic-20260924/results-review.md)と案Bの[独立受入審査](../../../target/i11-stage2-awake-diagnostic-20260924/acceptance-review.md)を受け、Astra監督下で次の事後限定受入を採用した。原登録試行の§5.7は11欄不合格のまま。以下の3段落を受入声明とする。

> **I11-2限定技術完了（追加証拠による事後受入）。** 今回のMac、封印済み候補binaryと登録入力、AC給電・蓋開放・system sleepなしを維持した追加試行という範囲で、代表身体footprintと実観測に由来する到来CDFをonset候補比較へ返す経路を技術受入する。数値核・配送/消費の強制fixture、Noneの旧12条件とoffの旧24条件の回帰、T1/T2の作用、gap分布を確認した。性能対象の旧8条件と新6条件、report有無の28組について、追加試行の全560性能欄と36適格性は取得前に固定した旧floorの規則に適合した。
>
> 原852本の試行では性能11欄が不合格であり、その判定は維持する。本受入は追加診断の結果を得た後の明示的な工学判断で、原試行の事前登録条件を満たしたという宣言ではない。許容値、raw hop、反復、失敗結果を変更・除外していない。追加試行にも137.379 msのhopが1件残り、元のfloorなし規則は239欄不合格だった。
>
> この受入は、給電状態やsleepが原失敗の唯一の原因だったという証明、統計的同等性、全電源状態での性能、無中断liveのdeadline適合、R2全資源受入、作者採用A1/A2、実機受入A3、O15/O18全体の達成を含まない。実kick等に条件付けたI10予測の消費も含まない。診断Hazard係数は未fitであり、既定Noneを維持する。
