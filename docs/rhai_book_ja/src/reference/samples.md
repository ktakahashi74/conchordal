# サンプル

12個の小さなサンプルを、順番に並べています。実行してscriptを読み、instrumentの主な機能を
一つずつ確認するためのものです。音楽作品ではなく、APIと挙動のデモです。

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

1. **A Single Voice** — 一つのVoiceが現れ、息を保ち、去る。
2. **Constellation** — line、peak、density、strain、gap、chanceという六つの入り方。
3. **Gravity** — 二つのsunの下の同じroot。peakはchartではなく現在の音を聴く。
4. **Tension** — Voiceが協和へ落ち着き、不安定になって離れ、再び冷えて落ち着く。
5. **Settling** — 散らばったVoiceが、Consonance Fieldに支えられる場所へ滑らかに移る。
6. **Bells** — 打撃されるbody。最後のbellはpartialをFieldに選ばせる。
7. **Heartbeat** — 外部の足場なしにPopulationが共有pulseへ同期する。
8. **Murmuration** — flockが命令されずに同期へ漂う。
9. **Rain** — beatを持たず、Fieldに沿って落ちる時間。
10. **Generations** — Voiceが生き、飢え、和声に支えられる場所へ生まれ直す。
11. **Autumn Cycle** — directionを持つharmony。季節が巡り戻る。
12. **Emergence and Resolution** — すべてを一つのarcへ曲げる。

サンプル1–6はConsonance terrain（placement、gravity、tension、movement、timbre）、
7–9はrhythm continuumを領域ごとに示し、10はloopをlifeとして閉じます。11–12は複数の
mechanismを組み合わせます。全サンプルはtest suiteでcompile-checkされ、現在のAPIとの
不一致を検出します。最上位のサンプルは意図的に`seed(...)`を呼ばず、毎runを新しい
scenario seedで始めます。
比較を再現可能にする必要があるresearch assayだけが固定seedを使います。

一つのcraft ruleが全体を貫きます：**scaffoldingは聞こえないか、embodyされる**。
terrain anchorは、drone自体が主題でない限りpresentationではなくhabitat busへ歌います。
pulse carrierには共鳴するbodyとlifeを与えるか、colonyからpulseが凝結するに任せます。

## Research assay

`samples/research/`にはheredity/selection ablation、external-scaffold rhythm control、
mechanism studyなどの比較fixtureがあります。instrumentを演奏するのではなく研究するもので、
上の道筋には含まれません。

## Offline rendering

`conchordal` instrumentはaudioをdiskへ書きません。演奏は、音が鳴り終わるとinstrument内には
残らない一過的な出来事（ephemeral）として設計されています。offline WAVには、同じcore engineを
共有する別binaryの`conchordal-render`を使います。
