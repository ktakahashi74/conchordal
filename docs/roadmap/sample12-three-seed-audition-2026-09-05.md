# 現行sample 12の3seed試聴 — 2026-09-05

専用pulseを除いた現行sample 12をseed 1・21・42で生成した。
3条件とも音源・reportの検証に成功した。作者は「ちらばったところからまとまってゆく流れは明確」と報告した。
続いて作者は、3seedすべてで同じ印象を持ち、どれも協和して終わると確認した。
散らばりからまとまりへの流れと協和した終わりを、3seed共通の試聴結果として記録した。
前回の音量A/B試聴は「差は聞き取れない／落ち着きは同程度」で記録済みである。
今回はflow減音を含む現行sample全体を対象とし、音量操作だけの比較とは区別する。

## 標準比較の操作窓を修正

`scripts/evaluate_beta.py`のsample 12集計には、専用pulseを除去する前の時刻が残っていた。
脚本の配置・操作順・waitの検査と操作窓を`scripts/sample12.py`へ集約し、標準比較、
8条件の操作分解比較、flow音量比較から共通の契約を使うようにした。
標準比較は、保存したsampleの構成が契約と異なればbuild・renderの前に停止する。
操作窓にListenerTwinの標本がない場合も失敗として記録し、空の集計を成功とは扱わない。

| 窓 | 開始秒（含む） | 終了秒（含まない） |
|---|---:|---:|
| baseline_colony | 2.3 | 11.7 |
| tension_before_flow | 11.7 | 15.0 |
| tension_with_flow | 15.0 | 20.3 |
| resolution_with_flow | 20.3 | 23.6 |
| resolution_after_flow | 24.9 | 30.2 |

修正対象は比較ツールであり、sample本体とRustエンジンの変更はない。
以前のcampaignは、そのときのsourceと操作窓を記録した成果物として保持する。

## 実行と検証

```bash
python3 scripts/evaluate_beta.py \
  --samples 12_emergence_and_resolution --seeds 1 21 42 \
  --reserve-runtime-ids-through 18 \
  --output target/resolution-evaluation/2026-09-05-sample12-three-seeds
```

- 3条件すべて成功。各音源は48 kHz・mono PCM16・36.213333秒で、全編無音ではない。
- PCMの飽和標本は3条件ともゼロ。
- 各seedの5窓に881・310・497・309・497個のListenerTwin標本があり、現行の境界時刻で集計した。
- source 177ファイル、設定、保存したreleaseバイナリ、各WAV・reportのハッシュを確認した。
- 設定とバイナリは前回の音量比較と同一。seed 1のWAVも前回の減音あり条件と完全一致した。
- Python検査39件成功。標準runner経由で5窓の境界と半開区間を検査し、脚本のwait変更ではrenderに進まないことを確認した。
- 通常Rust検査659件成功。全出力と終了値は`test_report.txt`・`test_status.txt`へ保存した。
- ブラウザーで3音源の再生、HTTP応答、携帯幅の表示を確認した。

| seed | PCM peak | PCM RMS | 作者の流れ・終わりの判定 |
|---:|---:|---:|---|
| 1 | 0.368011 | 0.043548 | 散らばりからまとまりへの流れは明確。協和して終わる |
| 21 | 0.371429 | 0.047038 | 散らばりからまとまりへの流れは明確。協和して終わる |
| 42 | 0.402985 | 0.046487 | 散らばりからまとまりへの流れは明確。協和して終わる |

全音源でゲイン変更や個別の音量正規化は行っていない。ListenerTwinのtensionは、
解放操作の前後でseed 1・42では低下し、21では上昇した。これは局所的な改善余地の指標であり、
作者の帰還感・終止感の判定には置き換えない。

## 試聴と次の判断

[3seed試聴ページ](../../target/resolution-evaluation/2026-09-05-sample12-three-seeds/index.html)をFirefoxへ開いた。
全編をseed 1→21→42の順に聴き、各seedについて次を記録するよう依頼した。

1. 落ち着く方向への変化が聞こえるか。ある場合はその時刻。
2. 終わり方を自然と感じるか、唐突か、判断できないか。
3. 他に気になった変化や区切り、その時刻。

作者の回答（2026-09-05）: 「ちらばったところからまとまってゆく流れは明確」

対象seedと終わり方への補足回答: 「どれも同じ  協和しておわるのもおなじ」

これにより、3seedすべてで散らばりからまとまりへの流れが明確に聞き取れ、協和して終わると確認された。
作者の表現をそのまま評価欄へ反映し、今回の3seed試聴と結果記録はこの2点について完了とする。
これは現行sample全体の流れに対する評価であり、flow減音だけの寄与を確認した結果ではない。
変化を感じた時刻は指定されていないため、ListenerTwinの特定の集計窓と同じ時刻の知覚判定としては扱わない。
報告は`author-listening.json`にも保存し、生成時の成果物とハッシュで結び付けた。
`listening.csv`の各seedの`tension_resolution`と`release`にも回答を記入した。他の評価軸は未判定のままとする。

提示した音源ごとに判断を残し、一つのseedの結果を他seedへ転記しない。
seed 1が前回と同一の音源でも、音量A/Bの判定だけで曲全体の受入判定を埋めない。
本campaignは現行sample本体の3条件であり、8条件の操作分解を3seedで再実行した結果ではない。

次はsample 07〜09の拍の可聴性、同期の立ち上がり、非拍節的なまとまりを評価する。

成果物は`target/resolution-evaluation/2026-09-05-sample12-three-seeds/`に保存した。
`manifest.json`、各`metrics.json`、`validation.json`から条件と機械検証を確認できる。
