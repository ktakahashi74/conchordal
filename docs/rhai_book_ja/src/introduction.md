# はじめに

[English version](/docs/rhai/)

Conchordalは、生成的な作曲のための生体音響楽器です。Rhaiは、この楽器にシナリオを
与えるための言語であり、音符を順番に並べるためのものではありません。音から
Consonance Fieldをつくり、複数のVoiceがその中を移動し、生き残り、世代を重ねながら
Populationを再編成します。

シナリオでは、まず**PopulationSpec**を定義します。それを`place()`で配置すると、
世代交代を通じて同一性を保つ**Population**になります。Populationを構成する一個の
生きた要素が**Voice**です。同じLandscapeを共有するすべてのPopulationを合わせたものを
**Community**と呼びます。和声は心理音響的なroughnessとharmonicityから生まれ、
リズムはCommunity自身が駆動する共有拍の上で創発します。スクリプトの役割は、音符を並べる
シーケンサーではなく、全体の条件と流れを整える演出者です。

Conchordal v0.4.0は、こうしたコンセプトを直接検討したい研究者と開発者のための
research alphaです。機能は未完成で、不安定な場合があります。作曲家や制作者は
beta版を待ってください。一般的な音楽制作の語彙で覆い隠さず、このモデル固有の
言葉で説明します。

## 本書の構成

- **チュートリアル**では、最初の音を鳴らし、エディタを設定し、演奏の実行、analysisの
  記録、再現を扱います：
  [クイックスタート](tutorial/quick_start.md)、
  [エディタ設定](tutorial/editor_setup.md)、
  [演奏](tutorial/observing.md)。
- **コンセプト**では、一つの生きたVoiceから、生態系と時間構造の全体までを
  組み立てます：
  [Population — Voiceをまとめる持続単位](concepts/voice_life.md)、
  [VoiceとLandscape — 音と環境のフィードバック](concepts/ecological_loop.md)、
  [Consonance Field — 音高を評価する地形](concepts/consonance.md)、
  [リズム](concepts/rhythm.md)、
  [ルーティングとListenerTwin](concepts/routing.md)、
  [タイムラインと構造](concepts/timeline.md)。
- **リファレンス**には、生成された[APIリファレンス](reference/api.md)と、
  順に試すための[サンプル](reference/samples.md)があります。APIは
  **Core API**、Core候補を試聴する**Experimental**、**Mechanism Tuning**、
  **Research Controls**の四層です。まずCoreから始め、作品が必要とするまでは
  残りを無視してください。登録signatureとtier分類はengineから生成して照合し、
  意味を説明する文章はdocumentation registryで管理します。

本書のすべての`rhai`コードブロックは、テストスイートによって実際の
スクリプトengine上で実行されます。これにより、現在のengineでcompile・実行
できることを確認します。ただし、説明文の意味まで自動的に証明するものではありません。
