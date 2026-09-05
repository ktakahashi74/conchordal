# beta 比較結果 — 2026-09-05

標準12条件のrelease実行と成果物検証がすべて成功した。07の単独pulseは発音間隔が安定した一方、
08の内部同期の立ち上がりと12の解放操作後のtensionにはseed差がある。音楽的な合否は未試聴のため未判定とする。

これは下記snapshotの測定記録である。後続の[Sample 12操作分解比較](resolution-assay-2026-09-05.md)で、
ListenerTwinの質量重み、小幅な音量操作、sampleのglide指定を修正した。以下の旧測定値は変更せず、修正版と区別して参照する。

## 成果物と再現条件

- [試聴ページ](../../target/beta-evaluation/2026-09-05-baseline-r2/index.html)、[試聴票](../../target/beta-evaluation/2026-09-05-baseline-r2/listening.csv)、[全条件CSV](../../target/beta-evaluation/2026-09-05-baseline-r2/summary.csv)
- [manifest](../../target/beta-evaluation/2026-09-05-baseline-r2/manifest.json)、[追加検証記録](../../target/beta-evaluation/2026-09-05-baseline-r2/validation.json)、[実行手順](beta-evaluation.md)
- 保存先: `target/beta-evaluation/2026-09-05-baseline-r2/`。生データはGit管理外のローカル生成物。

```bash
python3 scripts/evaluate_beta.py
```

基準commitは`390ae1e5c382c87dc1abae8bfd0673b3e778529d`と未コミット差分。実行時のsource 168ファイル、
設定、releaseバイナリとSHA-256をcampaign内に保存した。4作品×seed 1・21・42を逐次実行し、
各WAVとJSONLは同じrenderから得た。WAVは合計372.480秒、mono PCM16・48000 Hz。
解析hopは512 samples（約10.667 ms）、nfftは16384。DCC結合とHabituationは既定の無効設定。
全DCC記録の圧力が0であることも確認した。`CONCHORDAL_LIMITER`の環境上書きを除去し、設定ファイルを使用した。

測定機はAMD Ryzen 9 9950X、32 logical CPU、Linux 7.0.0-30-generic、Rust 1.95.0。
`cargo build --release --locked`、既定feature `simd-wide`、report有効。音声機器は開いていない。
自分のRust検証・ビルドを完了してから各測定を実行した。実機callbackのunderrunとメモリ割当数は未測定。

## 条件別の測定値

Kuramoto平均と同期開始は生成側のarticulation oscillatorの指標、beat confidenceはpresentation音声を分析したListenerTwinの指標である。
同期開始は内部orderが0.70以上で2秒続いた区間の開始を指す。可聴の拍や実際の発音位相の同期を直接表す値ではない。

| Sample | seed | onsets | Kuramoto平均 | 同期開始 秒 | beat confidence最大 | hop p95 ms | hop p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 07 | 1 | 488 | 0.938 | 7.531 | 0.687 | 1.598 | 1.714 |
| 07 | 21 | 488 | 0.913 | 6.016 | 0.608 | 1.604 | 1.719 |
| 07 | 42 | 485 | 0.988 | 3.381 | 0.860 | 1.564 | 1.755 |
| 08 | 1 | 462 | 0.520 | 19.627 | 0.489 | 1.777 | 1.960 |
| 08 | 21 | 465 | 0.687 | 3.499 | 0.464 | 1.801 | 1.880 |
| 08 | 42 | 448 | 0.801 | 10.336 | 0.578 | 1.634 | 1.870 |
| 09 | 1 | 702 | 0.733 | 7.723 | 0.404 | 3.994 | 4.202 |
| 09 | 21 | 732 | 0.628 | 2.624 | 0.488 | 4.037 | 4.318 |
| 09 | 42 | 714 | 0.336 | 未成立 | 0.433 | 3.959 | 4.078 |
| 12 | 1 | 676 | 0.989 | 2.304 | 0.994 | 2.793 | 2.992 |
| 12 | 21 | 681 | 0.938 | 2.304 | 0.900 | 2.768 | 2.876 |
| 12 | 42 | 674 | 0.950 | 2.304 | 0.875 | 2.826 | 2.981 |

先頭2秒を除いた各hopの分布からp95/p99を求めた。処理時間には分析待ちと先行するreport処理を含み、
`hop_timing`自身の書き込みを除く。p99は1.714〜4.318 msで、今回のオフライン条件ではhop時間以内だった。
これは4・16・64 Voiceの実機負荷試験の代わりにはならない。reportの有無による負荷差も未測定である。

全12条件でWAVとhop数が一致し、全編無音と量子化後の飽和標本はなかった。
ListenerTwinの`analysis_lag_frames`は全記録で1だった。この値はframe IDの差であり、解析窓・フィルタの遅延を含む総聴覚遅延ではない。

## 拍と同期の読み方

07のPopulation 2（beat）と12のPopulation 2（pulse）は単独Voiceなので、同時発音による0秒IOIの混入を避けて比較できる。

| Sample / Population 2 | seed 1 | seed 21 | seed 42 |
|---|---:|---:|---:|
| 07 / beat_stability | 0.976 | 0.973 | 0.991 |
| 12 / beat_stability | 0.987 | 0.982 | 0.981 |

- 07の単独beatの間隔指標は3seedとも高い。全体の`beat_stability=0`は多声部の同時発音を含む統計であり、拍の消失を意味しない。
- 08の内部同期開始は3.499〜19.627秒に分かれた。集団の立ち上がりを一つのseedだけで代表させない。
- 09のseed 42では内部同期の条件が成立しなかった。Rainは非拍節的な挙動を調べる作品なので、この値だけを失敗条件にしない。
- 12の同期開始2.304秒はpulse登場直後であり、この後2秒の`kuramoto_active_count`は1。colonyは6秒に入るため、2.304秒を集団同期の成立時刻として使わない。

## Sample 12の緊張と解放

以下は操作窓ごとのListenerTwin `tension_level`平均。窓はscriptの操作時刻から定めたもので、
聴き手が知覚する区間を確定したものではない。各窓の観測数とbeat confidenceは各条件の`metrics.json`に保存した。

| 操作窓 秒 | seed 1 | seed 21 | seed 42 |
|---|---:|---:|---:|
| colony追加後 6.0–15.4 | 0.00890 | 0.01285 | 0.01249 |
| tension操作後 15.4–18.7 | 0.01137 | 0.02418 | 0.01475 |
| flow追加後 18.7–24.0 | 0.01641 | 0.01661 | 0.01128 |
| resolution操作後 24.0–27.3 | 0.01999 | 0.00950 | 0.01722 |
| flow解放・colony減音後 28.6–33.9 | 0.01109 | 0.01362 | 0.01639 |

resolution操作後の24.0〜27.3秒のtension平均は、直前の18.7〜24.0秒と比べてseed 1と42で上昇し、seed 21だけが低下した。
後半の28.6〜33.9秒も、3seedとも6.0〜15.4秒の平均を上回った。現行の操作列が聴取モデル上で
一貫した緊張→解放を生むとは確認できない。音としての解放感が弱いか、モデルがその解放を捉えていないかは試聴と対照実験で分ける必要がある。

次の分解比較では、探索temperature、root/pulseの周波数移動、flowの追加・解放を一つずつ変え、
同じseedで寄与を調べる。音量・生存Voice数・無音尾部を併記し、減音だけを解決の成功としない。
新しいphrase/formモデルへの拡張判断は、この比較と作者の試聴結果を得てから行う。

## 計測の修正と検証

初回実行で、Coupled発音経路の未観測PLVが0として報告される不備を見つけた。
空の`SlidingPlv`は`None`、観測した0は`Some(0.0)`として区別し、JSONLの未観測値を`null`へ修正した。
標準12条件を修正後に再実行し、初回に対する全12WAVのSHA-256一致を確認した。
各作品の3seedのWAVは互いに異なる。全12条件の`mean_plv`は未観測の`null`となり、同期評価には使用しない。

Rustテスト643件、Pythonテスト10件、Clippy、全ターゲット検査が成功した。report有無によるWAV同一性、
report書き込み失敗の非0終了、JSONL欠落・不正値、WAV破損、timeout、既存成果物の保護を検証した。
試聴票は全項目未判定。実機での音切れ、対応Voice数、割当数、音楽的な合格は今回の成功件数に含めない。
