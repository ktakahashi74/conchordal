# Conchordal 現行アーキテクチャ正本

基準 revision: `origin/main` `dee5a91d`（2026-11-21 時点、`src/` 計 181,191 行 / 289 `.rs` ファイル）。  
本書は「いま動いている楽器」の構造だけを短く保つ正本である。設計経緯・陰性結果・詳細検証の記録は歴史アーカイブ（`docs/design-notes/technote-ledger.ja.md` 9,022 行、`docs/design-notes/dcc-neurocognitive-hierarchy.md` 5,961 行）に委ね、本書には追記しない。

---

## 1. 中核原則（変わらない契約）

1. **直接認知結合（DCC）**：音名・音階・拍子記号をエンジン内部に持たず、`Log2Space`（対数周波数軸）上の聴覚認知ポテンシャル場（R・H・C・慣れ）を環境として音を生成する。
2. **音だけで結合する（Only through sound）**：Population（Voice）・Flora・Rain は互いの内部状態を読まず、habitat バス上の実音と共通の聴覚モデルだけを介して相互作用する。
3. **根拠の3区分（台帳 §9.3.104）**：すべての機構を `Principle`（聴覚原理）／`Candidate`（公刊モデル候補）／`Author rule`（作者・作曲者規則）に分け、新規軸は既定中立（0）とする。
4. **Air-Gap**：楽器バイナリ `conchordal` と Web 版は音声・音声由来データを保存・送信しない。オフライン書き出しは別バイナリ `conchordal-render` が担う。
5. **単一コア共有**：`native`（GUI / cpal / WAV）と `web`（`StepEngine`・単一スレッド・ファイル/音声I/Oなし）が同じエンジンコアを共有する。

---

## 2. 1 hop の信号・制御フロー

```text
Rhai 台本 (finite: 一括評価 / endless: 逐次スレッド)
  └─▶ Conductor (時刻順 Action 配信)
        └─▶ Community (advance_population)
              ├── Populations / Voices  (離散個体: 移動・発音・代謝・再出生)
              ├── Flora                 (スペクトル植生: バイオマス力学・レーン合成)
              └── Rain                  (衝撃分布 + 共鳴物体)
                    │
                    ▼ render_and_route_audio (ScheduleRenderer)
              ┌─────┴─────────────────────────┐
              ▼ habitat bus (生態の耳)          ▼ presentation bus (聴き手の耳)
     [room=both 時は部屋適用 + LOO]      [room=presentation/both 時はステレオ部屋]
              │                               │
              ▼                               ├──▶ Limiter ─▶ AudioOutput / WAV / WebAudio
     AuditoryFrontend (因果複素FB, 1/4 ERB,   └──▶ ListenerTwin (提示専用解析 + E0 予測)
       位相再割当て, 臨界帯域圧縮, S6 密度)              │ (dcc.coupling_strength > 0 時のみ
              │                                          │  static_pull_level を温度へ還流)
              ▼                                          ▼
     Landscape (R, H, C, Habituation, Meter,   UiFrame (viewdata / viewdata_web) ─▶ GUI
       LOO/Flora/Tonal/Support 観測)
              │
              └─▶ 次 hop の Community へ還流
```

- **解析空間**：既定 48 kHz・hop 512 標本（10.667 ms）・約 20.015 Hz〜12.5 kHz（892 bin / 198 チャンネル）。NSGT は GUI・report・Flora 資源など実利用者がいる時だけ走らせる。
- **自己音除外（LOO）**：対象 Voice / Flora の実 habitat PCM を捕捉し、混合 habitat PCM から引いて聴覚前段へ通す（`src/core/stream/loo.rs`）。
- **部屋（`[room]`）**：`off`（D/D・モノラル、既定）、`presentation`（D/W・ステレオ 2ch、Jeub 2011 両耳コヒーレンス）、`both`（W/W・研究用、個体別 strict 自己除去）。`site` により音源距離（12 m 聴衆基準）を設定可能。

---

## 3. 3つの住人（発音系）

| 住人 | 主モジュール | 役割 | 音の作り方 |
|---|---|---|---|
| **Population / Voice** | `src/life/voice.rs`, `pitch_core.rs`, `phonation_engine.rs`, `sound/` | 輪郭を持つ離散個体。協和地形を探索し、エネルギー代謝・寿命・再出生を行う | `Sine` / `Harmonic` / `Modal`（`ScheduleRenderer` 経由） |
| **Flora** | `src/life/flora.rs`, `flora_metric.rs`, `flora_respond.rs` | 地形上で育ち、資源を消費し枯れるスペクトル植生（和声の面・床） | トーンレーン＋残差帯域を直接合成 |
| **Rain** | `src/life/rain.rs`, `sway.rs` | 1/f 揺らぎを持つポアソン衝撃列と、打撃で鳴る共鳴物体 | 衝撃スロット（64）＋モーダル物体（最大16） |

---

## 4. モジュール責務と規模（`origin/main` `dee5a91d`）

| モジュール | 総行数（テスト含） | 責務 | 現在の主な課題 |
|---|---:|---|---|
| `src/life/` | 76,084 | Voice・Flora・Rain・Community・発音・レンダラ | `pitch_core.rs` / `voice.rs` / `flora.rs` の肥大、`Voice`/`Flora` の40超フィールド、`RefCell` |
| `src/core/` | 34,606 | 聴覚前段・R/H/C・NSGT・慣れ・拍・LOO・部屋 | `Landscape` の約45フィールド同居、`auditory_frontend.rs` 内の `#[cfg(test)]` 散在 |
| `src/temporal_cognition/` | 23,165 | 筋・群・身体観測・到着・和声時間関係・opt-in T2 | 既定オフの研究・拡張コードの占有が大きい |
| `src/scripting/` | 19,764 | Rhai エンジン・186 API・型定義/ドキュメント自動生成 | 既定0の Experimental/Tuning/Research 語が堆積 |
| `src/runtime/` | 13,863 | スレッド配線・worker loop・`StepEngine`・予約・監視 | `WorkerState` の40超フィールド、`#[cfg(test)]` フック |
| `src/ui/` | 2,750 | egui 描画（`windows.rs`, `plots.rs` のみ） | — |
| `src/listener_twin/` | 2,292 | 提示専用聴き手モデル・E0 予測 | — |
| `src/audio/` | 1,636 | cpal 出力・Limiter・WAV ライタ・復旧 | — |
| `src/scenario/` + `scenario.rs` | 2,300+ | シナリオ IR・制御設定・ライフサイクル設定 | — |
| `src/synth/` | 949 | モーダル共振器バンク・モード表 | 実合成が `src/life/sound/` 等へ分散 |
| `src/viewdata*.rs`, `config.rs` 他 | 4,767 | UI 契約・TOML 設定・クロック・handoff | `config.rs` に研究用 opt-in 構造体が多数同居 |

---

## 5. 既定で有効なもの vs 任意・研究のもの

- **既定で有効（楽器のベースライン）**：
  - 聴覚前段（因果複素FB + S6）→ R / H / C → 共有 `HabituationField`（`channel_p`）
  - Population（Voice の移動・発音・代謝・再出生）・Flora・Rain
  - 提示専用 `ListenerTwin`（GUI 時 E0 予測表示、生成へは戻さない）
  - 部屋 `off`（デモ・明示指定時は `presentation` ステレオ）
- **既定 0 / opt-in（明示指定または研究設定時のみ有効）**：
  - `[dcc].coupling_strength`、`[room] mode = "both"`、`felt_beat` / `syncopate`
  - 候補層（TC・PS-1・SQ・ME・PX・PE・`listener_profile("tonal")`）
  - 和声拡張（`support_*`・`harmonic_relations`・`harmonic_rhythm`・`tail_roughness_aversion`）
  - `temporal_meter_t2`、7種の `temporal_*` 診断設定、`render_prototype`

---

## 6. 構造変更時の不変条件（必須ゲート）

1. **決定性**：`native` の各 SIMD 段とスカラーで bit 一致（`web` は `native` との bit 一致不要、有限出力・underrun 0 が条件）。
2. **ゼロ重み不変**：既定 0 の機構は無効時に乱数消費・PCM を 1 bit も変えない。
3. **実時間予算**：定常 `worker_loop` と cpal コールバックで新規ヒープ確保を増やさない（48 kHz / hop 512 = 10.667 ms 予算）。
4. **Rhai 表面の整合**：`src/scripting/docs.rs` と `engine.rs`、`rhai-defs/conchordal.d.rhai`、日英 API リファレンスは常に完全一致（CI 検査）。
5. **依存境界**：`runtime/` は `ui/` に依存しない（`viewdata.rs` のみ経由）。`conchordal` バイナリにディスク書き出し経路を持たない。
