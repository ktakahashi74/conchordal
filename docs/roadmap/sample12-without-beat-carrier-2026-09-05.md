# Sample 12から専用の拍打ちを除去 — 2026-09-05

作者はsample 12の専用pulseを音楽的でないと評価し、明示的な同期が必要なsample以外では使わない方針を示した。
この方針を[AGENTS.md](../../AGENTS.md#scenario-script-authoring)へ反映した。
非可聴のhabitatへ移すだけでも同期補助は残るため、sample 12ではVoice自体を除去した。

## Sampleの確認

- 公開12 sampleのうち、専用の拍打ちVoiceがあったのは07と12である。
- 07 Heartbeatは明示的な同期を実演するsampleのため、downbeatを維持し、その理由をコメントへ追記した。
- 12では`metric_body`の定義、`pulse`の配置・音高操作・release、pulse専用の待ち時間を削除した。
- 音高の基準を作るhabitat専用drone、colony自身の`.metric()`、flowは維持した。
  専用pulseをなくす変更であり、周期性のない音を保証するものではない。
- `research/temporal_scaffolding_*`は同期条件を比較する実験であり、明示的なscaffoldを維持した。
  他の研究用sampleに、音楽的な支えとして追加された専用の拍打ちVoiceは見つからなかった。
- 日英Rhai bookのrhythm章にあるbeat Voiceは、同期の説明に用いる例として維持した。

## 構成と比較の更新

現行版のPopulationはroot=1、colony=2、flow=3である。
Voice ID予約はroot 1 + colony 8 + flow 9の18へ変更した。
sample本体にseedは追加していない。

| 操作 | pulseあり版 | 現行版 |
|---|---:|---:|
| colony登場 | 6.0秒 | 2.3秒 |
| 探索加熱・root上昇 | 15.4秒 | 11.7秒 |
| flow登場 | 18.7秒 | 15.0秒 |
| 探索冷却・glide・root帰還・flow減音 | 24.0秒 | 20.3秒 |
| flow解放 | 27.3秒 | 23.6秒 |
| colony減音 | 28.6秒 | 24.9秒 |
| colony解放 | 33.9秒 | 30.2秒 |
| root解放 | 37.2秒 | 32.2秒 |
| section終端 | 41.2秒 | 36.2秒 |

`scripts/evaluate_resolution.py`の操作断片、配置数、操作時刻の検査、Population抽出、ID予約、
出力文言を新しい構成へ合わせた。flow有無の比較は介入前[0,15)秒のPCMと状態記録の一致を要求する。
Population 3のflowをcolonyとして集計しない回帰条件も確認する。

旧campaignの音源・数値・作者の試聴記録は保存時のsourceに対する記録として保持する。
pulseの除去は生態系への入力と乱数の消費も変えるため、旧音源からpulseだけを引いた音にはならない。
新構成の試聴ページを案内した後、作者から「確認した。 commit」との返答を得た。
今回の変更は作者確認済みとする。音量の寄与や終止感についての追加判定は示されていない。

## 検証

- `cargo test -- --nocapture`：659件成功。公開sampleのコンパイル、研究用sampleのseed規則を含む。
  全出力と終了値は`test_report.txt`・`test_status.txt`に保存した。
- Python比較runner：35件成功。flowのPopulation 3をcolonyの集計から除く条件も通過した。
- 日英Rhai bookのsample案内とrouting説明を更新した。コード例・章対応・日本語規約の4検査が成功し、両bookをbuildした。
  検証用の生成物は`target/sample12-docs-validation/en/`・`ja/`に保存した。

```bash
python3 scripts/evaluate_resolution.py --seeds 1 \
  --output target/resolution-evaluation/2026-09-05-no-beat-carrier
```

seed 1の8条件が成功した。flow有無4組すべてで、登場前[0,15)秒のPCMと状態記録が一致した。
生成時のsource 174ファイル、設定、実行バイナリ、各条件の音源・reportのハッシュも確認した。
spawn記録はroot 1 Voice、colony 8 Voice、flowありの場合だけ9 Voiceであり、専用pulseのPopulationはない。
全8条件で全編無音ではなく、PCMの飽和標本はゼロだった。原本に対応する111の長さは36.213秒である。

- [専用pulseなしの試聴](../../target/resolution-evaluation/2026-09-05-no-beat-carrier/audition.html)
- [8条件の一覧](../../target/resolution-evaluation/2026-09-05-no-beat-carrier/index.html)
- [検証記録](../../target/resolution-evaluation/2026-09-05-no-beat-carrier/validation.json)
- [介入前の一致](../../target/resolution-evaluation/2026-09-05-no-beat-carrier/flow_pre_intervention.json)

今回のrender確認はseed 1だけであり、3seedの再評価は未実施である。
オフライン生成の成功を、実機性能や音楽的な受入判定へ読み替えない。
