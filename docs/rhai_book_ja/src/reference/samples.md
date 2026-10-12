# サンプル

16本のサンプルを、同じ`samples/`内の二つの群として紹介します。1〜12は、機能を
一つずつ確かめる小さな実演です。音楽作品ではなく、APIと挙動を示します。
13〜16は、それらの機能を組み合わせて曲として書いた、やや長い場面です。
どちらも実行し、scriptを読んで確かめられます。

```bash
cargo run --release -- samples/01_a_single_voice.rhai
```

## 機能の実演（1〜12）

1. **A Single Voice** — 一つのVoiceが現れ、息を保ち、去る。
2. **Constellation** — line、peak、density、strain、gap、chanceという六つの入り方。
3. **Gravity** — 二つのsunの下の同じroot。peakはchartではなく現在の音を聴く。
4. **Tension** — Voiceが協和へ落ち着き、不安定になって離れ、再び冷えて落ち着く。
5. **Settling** — 散らばったVoiceが、Consonance Fieldに支えられる場所へ滑らかに移る。
6. **Bells** — 打撃されるbody。最後のbellはpartialをFieldに選ばせる。
7. **Heartbeat** — 拍を打つ専用Voiceを使い、同期を実演する。
8. **Murmuration** — 指揮者のいない群れ。各Voiceは自分の間合いを保ち、聴こえた再来へ周期を寄せる。共有の拍は狙わない。
9. **Rain** — beatを持たず、Fieldに沿って落ちる時間。
10. **Generations** — Voiceが生き、飢え、和声に支えられる場所へ生まれ直す。
11. **Autumn Cycle** — directionを持つharmony。季節が巡り戻る。
12. **Emergence and Resolution** — Populationの探索が強まり、再び落ち着く。専用の拍打ちは置かず、Voice自身の発音がリズムの入力を担う。

## 機能を組み合わせた曲（13〜16）

13. **Three Windows** — 三本の線が音高を採用するたびに発音し、それぞれの扉から去る。
14. **Stone and Wire** — 拍を刻む二つのVoiceの間を、独立した線が横切る。
15. **Folded Air** — 相手が休みを置いて戻り、区間ごとに発音の形を変える。
16. **Open Court** — やや密な群れが、独立した二本の道の周りへ集まる。

## 道筋と実行

1〜6はConsonance Field上の配置、引力、緊張、移動、音色を紹介します。
7〜9は三つの時間の取り方を示します。`metric()`は音から確認された共有拍へ寄り、
`entrained()`は共有拍の位相を狙わず、音を聴いて参加します。`flow()`は拍を作らない、
かたまりの更新を担います。10は生存と世代交代の循環、11〜12は複数の機構の組み合わせです。
13–16は、採用ごとに発音する線、拍を刻むVoiceと横切る線、
休む相手と区間ごとの発音の切り替え、やや密な群れを示します。
全サンプルはtest suiteでcompile-checkされ、現在のAPIとの
不一致を検出します。最上位のサンプルは意図的に`seed(...)`を呼ばず、毎runを新しい
scenario seedで始めます。
比較を再現可能にする必要があるresearch assayだけが固定seedを使います。

音高の基準を作るanchorは、通常はhabitat busだけへ送られ、直接は聞こえません。
Gravityでは、聞こえる持続音そのものが主題です。Heartbeatは同期を実演するため、拍を打つ
専用Voiceを置いています。他の公開サンプルには専用の拍打ちVoiceはありません。

## Research assay

`samples/research/`にはheredity/selection ablation、external-scaffold rhythm control、
mechanism studyなどの比較fixtureがあります。instrumentを演奏するのではなく研究するもので、
上の道筋には含まれません。
専用の同期補助は、明示的な同期が実演や実験に不可欠な場合だけ使います。
habitat busだけへ送って聞こえなくしても、他のサンプルで使える理由にはなりません。

## Offline rendering

`conchordal` instrumentはaudioをdiskへ書きません。演奏は、音が鳴り終わるとinstrument内には
残らない一過的な出来事（ephemeral）として設計されています。offline WAVには、同じcore engineを
共有する別binaryの`conchordal-render`を使います。
