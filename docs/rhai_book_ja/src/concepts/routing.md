# ルーティングとListenerTwin

## 二つのbus

各々のVoiceは、独立した二つのモノラルバスへ寄与します。

- **presentation bus** → cpal出力 / offline render / UI metering（提示される作品）
- **habitat bus** → NSGT解析 → Landscape（ALife生態系が応答するもの）

初期状態では両方へ送られます。一方だけに送るには`send()`、両方を明示するには`|`で
結合します。

```rhai
// Reference anchor: sensed by the ecology, absent from the presented sound.
let anchor = harmonic().brain("drone").send(habitat_bus);

// Presented decor that does not influence the ecology.
let decor = sine().send(presentation_bus);

// Explicitly both (the default).
let normal = harmonic().send(habitat_bus | presentation_bus);

place(anchor, at(110.0));
place(decor, at(880.0));
place(normal, consonance(110.0).peak().count(3));
wait(4.0);
```

バスの分離は作曲の道具です。`send(habitat_bus)`の隠れたanchorは聞こえずにPopulationの
組織化を変え、`send(presentation_bus)`のdecorは生態系を乱さずに聞こえます。

初期状態ですべてのVoiceを両方のバスへ送るのは、それがDirect Cognitive Couplingだからです。
聴き手が聴くものと生態系が知覚するものが同じ物理eventになります。分離は意図的な逸脱です。
`send(habitat_bus)`は聞こえずにFieldを形づくるterrain、`send(presentation_bus)`は
聞こえるが生態系の世界の外にあるdecorです。サンプルでは、scaffoldingは聞こえないか、
Voiceとしての発音体と生を与えられます。

## ListenerTwin

Conchordalは、**提示された音だけ**をmodel化するlistener側の`ListenerTwin`を持ちます。
habitat busは読まないため、hidden scaffoldが偽のlistener tensionを作ることはありません。
`listener_state` reportには四つの知覚levelがあります。

- `stability_level` — 現在聞こえる音の安定性 / consonance。
- `resolvability_level` — 近くに、より安定したもっともらしい継続状態があるか。
- `tension_level` — `(1 - stability) * resolvability`。現在は不安定だが改善経路がある状態。
- `attention_level` — presentation由来のonset / spectral-flux salience。

listener側のmeter推定も含みます。

- `beat_hz`、`beat_phase`、`beat_confidence`
- `subdivision_ratio`、`subdivision_confidence`
- `measure_hz`、`measure_ratio`、`measure_confidence`

`generated_frame_id`、`analysis_frame_id`、`analysis_lag_frames`は知覚遅延を明示します。
report eventと原因となった音を比較するときに重要です。

Twinを操作するscripting verbはありません。命令する対象ではなく観測する対象です。
report有効時に`listener_state` recordを出し、GUIにも同じstateを表示します。generationへ
結合する前に、自分が聴くtensionとTwinのreportが一致するかを確認してください。

## DCC：Twinを戻す結合

任意の結合である**DCC**はscriptではなく`config.toml`（または`--config`で指定したfile）で
設定します。

```toml
[dcc]
# Listener pressure is report/UI-only by default.
# coupling_strength = 0.0
# max_temperature_bonus = 0.10
```

- `coupling_strength`（`0.0`–`1.0`、既定値`0.0`）：`0.0`ではレポートとUI専用で、
  generationは変わりません。値を上げると
  `tension_pressure = tension_level * resolvability_level * coupling_strength`を
  一時的な音高探索の加算値としてだけ使います。目標音高やリズム同期を
  直接設定しません。
- `max_temperature_bonus`（既定値`0.10`）：一時的な加算値の上限。

`listener_state`が音楽的に読めることを確かめてから`coupling_strength`を少しずつ上げます。
結合中は`dcc_pressure` recordにpressureとtemperature bonusが記録されます。
