# Population — Voiceをまとめる持続単位

Conchordalを理解するための最も簡単な捉え方は、音符の次に別の音符が来るというものでは
ありません。PopulationSpecが、生きたVoiceからなる持続的なPopulationへ変わります。

```text
PopulationSpec + Placement --place()--> Population --> Voice(s)
                                                |
                                                `-- later generations

Community = runtime terrainを共有するすべてのPopulation
```

**PopulationSpec**は再利用可能な配置前の定義です。初代のVoiceの初期値と、
生存過程、生存可能性、respawnなど、Population全体に属する方針を組み合わせます。
**Placement**は初代のVoiceがどこへ何体入るかを指定します。`place()`は両者を結合し、
ただちに**Population**を返します。これはVoiceごとの参照ではなく、配置されたPopulationを
表す一つの安定した参照です。実行時の**Community**は、Landscapeを共有するすべての
Populationを集約したものです。

```rhai
let population_spec = harmonic()
    .amp(0.035)
    .sustain()
    .respawn_capacity(6);

let population = place(
    population_spec,
    consonance(90.0, 900.0).count(6)
);
wait(3.0);
release(population);
```

ここで`population`は6つの初代のVoiceをまとめて制御します。レポートでは共有する
`population_id`と、各々のVoiceの`voice_id`を区別します。Voiceが死亡して代わりのVoiceに
置き換わると`voice_id`と`generation`は変わりますが、Populationと`population_id`は
変わりません。

## 配置境界

`place()`だけが定義から実行時への遷移です。初期設定専用の属性は、先に
`PopulationSpec`へすべて設定します。配置後の`Population`が公開するのは実行中の更新と
releaseだけで、specificationへ戻したりfounder policyを書き換えたりはできません。

```rhai
let spec = harmonic()
    .brightness(0.4)
    .brain("entrain")
    .endurance(8.0);

let population = place(spec, consonance(220.0).count(3));
population.amp(0.03); // Live patch at the current script time.
flush();              // Emit pending live patches without advancing time.
wait(2.0);
release(population);
```

`wait(seconds)`も保留中の更新を発行してからスクリプト上の時刻を進めます。
`wait()`も`flush()`も、未配置のPopulationを作りません。初代のVoiceは
`place()`そのものによってschedule済みです。[APIリファレンス](../reference/api.md)では、
`PopulationSpec`のメソッドを初期設定専用、`Population`のメソッドを実行中に更新可能と表示します。

## 独立した五つの問い

PopulationSpecは、複数の独立した問いに答えます。分けて考えることで、ある音楽的判断を
別の判断と取り違えずに済みます。

| 問い | 主なcontrol | 意味 |
|---|---|---|
| 何が鳴るか | `sine`, `harmonic`, `modal`, `brightness`, `modes` | 初代のVoiceの発音体と周波数成分。 |
| どんな生を送るか | `brain(name)`: `entrain`, `seq`, `drone` | articulationが生態系に参加するか、書かれた生を進むか、Landscapeのanchorとして持続するか。 |
| いつ鳴るか | `sustain`, `repeat`, `metric`, `entrained`, `flow` | phonationとonset timing。 |
| 音高はどこへ行くか | Placement、`anchor`, `seek_consonance`, `temperature` | 初代のVoiceが入る位置と、その後の移動。 |
| Populationはどう持続するか | `endurance`, `recovery`, viability, respawn | Voiceのエネルギー、死、Populationの世代交代。 |

別の行に属するcallは合成できます。同じ軸では通常、最後の指定が優先されます。正確な
構築時に各メソッドがどう働くかはAPIリファレンスに記載されています。

## Articulation life：`brain`

`brain(name)`は、鳴っている間にVoiceがどのような生を送るかを選びます。

- `brain("entrain")`は既定の、生きた発音様式です。代謝と生存過程の設定を
  通じてconsonanceとrhythmic fitに応答できます。
- `brain("seq")`は固定された生を持つauthored eventです。Field viabilityとmetabolismを
  無視します。
- `brain("drone")`は明示的にreleaseされるまで死にません。terrain anchorなど、
  持続するmaterialに適します。

これはphonation timingとは別です。

> `brain("entrain")`は生の種類を選び、`.entrained()`は反復onsetを共有meterへ
> 中程度に結合します。

似た名前ですが別の軸なので、同時に使用できます。

```rhai
let colony = harmonic()
    .brain("entrain")
    .entrained()
    .cycles(2)
    .seek_consonance()
    .endurance(8.0)
    .recovery(4.0)
    .consonance_viability(0.30, 0.80);

place(colony, consonance(90.0, 900.0).count(5));
wait(8.0);
```

`brain()`は発音体、Placement、音高の方針を選びません。droneは聴取可能にもhabitat-onlyにも
でき、生きたVoiceは固定も移動もできます。同じ共鳴発音体に任意のarticulation lifeを
組み合わせられます。

## Phonationとduration

Phonationは、onsetがいつ起きるか、そのonsetがどれだけ開いたままかに答えます。

- `sustain()`はVoiceが生きている間、音を保持します。
- `repeat()`は初期値が設定された反復発音を選びます。
- `metric()`、`entrained()`、`flow()`は共有meterとの結合連続体上の領域を選び、
  re-attackを含意します。
- `cycles(n)`はdurationをrhythmic cycle数で表します。

presetで意図を表せない場合は、低水準の`once()`、`pulse(rate_hz)`、`while_alive()`、
adaptive duration controlを使えます。まずpresetから始め、作品が必要とするときだけ
明示的timingを使ってください。

## Releaseはdeathではない

`release(population)`はスクリプトによる終端判断です。以後そのPopulationへの更新は無視され、
現在のVoiceはrelease envelopeへ入り、Populationは閉じます。生態学的な死は実行時の
結果です。一つのVoiceがenergyを使い果たした後、Populationの再生成方針が代わりのVoiceを
作る場合があります。`section`や`play`の有効範囲も、終了時に内部で作ったPopulationをreleaseします。

次章では、[VoiceとLandscape — 音と環境のフィードバック](ecological_loop.md)を説明します。
