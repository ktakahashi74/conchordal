# 同期した高密度発音の10分試験 — 2026-09-05

状態: この条件の600秒測定が成立。発音回数・間隔、性能、releaseを確認した。

[同期負荷の30秒試験](synchronized-load-2026-09-05.md)と同じharmonic・64 Voiceを、
600秒間測定した。以前の適応時計による約3回/秒の10分記録とは別の試行であり、
各Voiceが8回/秒の発音回数と間隔の条件を維持したことを確認した。

## 条件と手順

- AMD Ryzen 9 9950X、MOTU M2、48 kHz、stereo、hop 512（予算10.667 ms）、ring 4800 mono frames。
- harmonic 8 modes、64 Voice、seed 1、report有効、DCC 0.25、warmup 5秒、測定600秒、release後2秒。
- 16 Hzの共有θ scaffold、`pulse(32)`の蓄積率、既存の2ゲート間隔制限を使用する。
  同期が必要な性能アッセイであり、専用の拍打ちVoiceを追加しない。音楽的な創発の評価には使わない。
- 先行試験と同じ振幅、envelope、周波数配置を使用し、測定時間だけ延ばす。
- release・`profile-alloc`ビルドを使用する。実行中に自分のビルド・テスト・別の負荷試験を並走させない。
  機器設定、プロセス優先度、CPU governorは変更しない。
- 楽器`conchordal`で再生し、音声ファイルは保存しない。

```bash
python3 scripts/evaluate_rt.py --mode device --workload dense --timing synchronized \
  --voices 64 --bodies harmonic --reports 1 --dcc 0.25 --seed 1 \
  --warmup-sec 5 --duration-sec 600 --timeout 750 \
  --output target/rt-evaluation/2026-09-05-sync-long-harmonic
```

測定窓は生成時刻の`[5, 605)`秒とし、その中に完全に含まれるhopを評価する。
全hopで64 Voiceを維持し、callbackの出力不足ゼロ、callbackエラーゼロ、hop p99が予算以内であることを要求する。
各Voiceの発音回数は4800回に対して境界の1発分以内、発音間隔は125 msに対して1 hopと
時刻丸め誤差以内とする。onsetの時刻はreportのhop時刻であり、sample精度ではない。
同一Voiceの同一hop内の重複発音、共有scaffold情報の欠落、profileの欠落・打ち切りも拒否する。

10分全体に加え、1分ごとの処理時間・出力不足・発音を調べ、終了時のreleaseと全Voiceの消滅も確認する。
MOTU M2への接続とRSSは5秒ごとに別途観測する。RSSにはreport・profileの保持分が含まれるため、
増加量だけからメモリリークとは判定しない。接続の観測も連続監視ではない。

## 結果

| 指標 | 測定値 |
|---|---:|
| 測定対象hop | 56,249 |
| 生存Voice数 | 全対象hopで64 |
| hop p99 | 1.592 ms |
| 最大hop | 2.823 ms |
| 予算10.667 msを超えたhop | 0 |
| 測定窓の出力不足 | 0 frames |
| callbackエラー | 0 |
| onset総数 | 307,200 |
| 平均発音数 / Voice / 秒 | 8.000 |
| 最大同時Tone数 | 64 |
| worker割当 / hop | 116.758回、49,303.184 bytes |

全64 Voiceが600秒全体の発音回数・間隔検査を満たし、1分ごとの窓でも同じ検査を満たした。
全体のVoice内IOIは117.300〜128.060 ms、平均124.999 msで、report時刻の量子化を含む許容内に収まる。
runnerの`status=ok`と`device_rt_pass=true`を確認し、保存したprofileとreportから同じ判定を再計算した。

| 測定開始からの分 | hop p99 ms | 最大hop ms | 出力不足 frames | 発音数 / Voice / 秒 |
|---|---:|---:|---:|---:|
| 1 | 1.534 | 2.823 | 0 | 8.000 |
| 2 | 1.603 | 2.159 | 0 | 8.000 |
| 3 | 1.564 | 2.210 | 0 | 8.000 |
| 4 | 1.598 | 2.040 | 0 | 8.000 |
| 5 | 1.600 | 1.956 | 0 | 8.000 |
| 6 | 1.634 | 2.813 | 0 | 8.000 |
| 7 | 1.600 | 2.109 | 0 | 8.000 |
| 8 | 1.576 | 2.003 | 0 | 8.000 |
| 9 | 1.560 | 2.139 | 0 | 8.000 |
| 10 | 1.602 | 2.018 | 0 | 8.000 |

## 終了処理と計測の境界

605秒のrelease予定に達する最初のhopは605.0027秒で、処理時間は0.835 msだった。
605.04535秒に生存Voice数が0となり、605.056秒にrendererのTone数も0となった。
その後の尾部を含め、生成処理の最後のhop（607.008秒）まで出力不足は0だった。

プロセス終了前のcallback累積値には269,824 framesの不足が記録された。
これは最後の生成hopより後に増えた値である。コードでは生成ループを抜けた後にreportの最終集計とflushを行い、
その後のprofile書き出し時にcallback累積値を取得する。測定窓とrelease中の不足へ混ぜず、
`last_hop_underruns=0`と`final_callback_underruns_including_summary_time=269824`を別々に保持した。
全プロセスのcallback累積値が0だった、という結果ではない。

## 保存と限界

MOTU M2へのactive linkを121回観測した。最初と最後の観測間は601.144秒、最大間隔は5.014秒で、
取得エラーはなかった。RSSは43.594〜79.621 MiBだった。これはreport・profileの保持と終了時集計を含む
プロセス全体の観測であり、時間無制限のメモリ安定性を保証する測定ではない。

バイナリのSHA-256は`a8802f17bd956e4d2203dfc78d2b607daa96e353f5ced112acb34ea857fee525`。
先行する30秒試験とsource 177ファイル、バイナリ、設定が一致し、脚本は測定時間だけが異なることを確認した。
今回の測定では生成コードを変更していない。検証時点の作業ツリーの測定対象sourceもsnapshotと一致した。

保存先:

- 実行条件・source・バイナリ・profile・report・接続観測:
  `target/rt-evaluation/2026-09-05-sync-long-harmonic/`
- hash検査、全体と各1分の再集計、release記録、接続・RSSの集計:
  `target/rt-evaluation/2026-09-05-sync-long-validation/validation.json`
- 再検査スクリプト:
  `target/rt-evaluation/2026-09-05-sync-long-validation/validate.py`

この1音色設定・1seed・600秒の結果を、他の音色、任意のVoice数、無期限の演奏へは一般化しない。
同期高密度の長時間性能という予定した確認は完了した。次は小節アクセントの生成を扱い、
固定scaffoldの同期を集団自身から生じた音楽的な強弱と取り違えない。
