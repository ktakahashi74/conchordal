# 合成法と不可視次元に関する六つの断定の点検

日付: 2026-09-26。F0監査、現行解析経路、下記の原著論文・著者資料を照合した短い理論メモ。新しい合成器の実装や聴取実験の結果ではない。

| 元の断定 | 修正が必要な範囲 |
|---|---|
| 1. 自励発振はFletcherのmode lockingで常に整数倍音となり、加算合成と等価。残る価値は閾値・ヒステリシスだけ | **常に**を撤回。Fletcherの原論文はmode lockingに近い小整数比、十分な結合・非線形・振幅などの条件を挙げ、離れたモードでは非調和・うなりを生む場合も論じる。固定された周期的定常波形はフーリエ級数で加算表現できるが、圧力や結合の変化に応じた位相・振幅・周波数の推移、過渡応答まで同じ入力出力系になるわけではない。閾値・ヒステリシスは可能な差の一部。[Fletcher 1978 原論文](https://www.phys.unsw.edu.au/music/people/publications/Fletcher1978.pdf)、[Fletcher 1993 著者解説](https://phys.unsw.edu.au/music/people/publications/Fletcher1993c.pdf)。 |
| 2. 亜臨界Hopfは標準Stuart–Landauで十分に置換できる | 三次のHopf正規形は**局所的な亜臨界の不安定周期枝**を表せる。一方、通常の飽和型三次Stuart–Landauをそのまま使うと超臨界側であり、静止状態と安定な有限振幅発振の共存・ヒステリシスをそれだけで保証しない。例えば振幅方程式 `ṙ = μr + br³ − cr⁵`（`b,c>0`）は有限振幅の安定枝を持つ。五次項や別の飽和機構を指定し、元の励振・モード結合のどの性質を残すか別に検証する必要がある。[Freyerほか 2012 原著](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002634)。 |
| 3. Pressnitzer–McAdams 1999の位相依存roughnessはpower spectrum分析では全面不可視 | 同じ振幅スペクトルでも知覚roughnessが変わったという実験結果から、**その振幅スペクトルだけ**を入力とする静的関数は該当刺激の差を識別できない。著者らは物理的波形包絡が同じでも差が出る条件と、包絡形状が変わる条件を分け、聴覚末梢処理後の内部包絡と時間的非対称性を論じる。「powerを使う解析の全経路が差を一切保持しない」は別の命題。現行のNSGTはhopごとのpower履歴を持ち、DorsalはPCMを直接処理する。両経路が原実験の位相効果を再現するかは未測定。[Pressnitzer・McAdams 1999 原著](https://www.mcgill.ca/mpcl/files/mpcl/pressnitzer_1999_jasa.pdf)。 |
| 4. 20–55 Hz pulseは55 Hz下限なら全て分析外 | 55 Hzは現行H/R用Log2Spaceの中心周波数下限。20–55 Hzの純音の基音はその格子に直接載らない。一方、鋭いpulse列は55 Hz以上の成分と時間変動を持ちうる。現行DorsalはPCMから約200 Hz以下を含む3帯域energyと正のfluxを計算し、production meterはfluxとphonation onsetを受ける。「全解析外」は誤り。どのpulse条件をどの経路が弁別するかは別途測定が必要。根拠: `src/runtime/mod.rs`の解析空間・`drive_production_meter`、`src/core/stream/dorsal.rs`。 |
| 5. 固定包絡でtessitura gravityを代替できる | 現行gravityは候補基音のlog2位置と個体の中心との差に対する明示罰則 `g(log2 f−c)²`。固定スペクトル包絡は音の周波数別強度を変え、特定条件では似た音域バイアスを作れるが、個体ごとの中心・移動費用・他個体との相互作用と一般に等価ではない。代替と主張するなら、同じ候補・場面で順位と動態を比較する。根拠: `src/life/pitch_core.rs`の`gravity_penalty`、[身体適応度計画](body-aware-fitness-plan.md)の固定／移動包絡を分ける検査。 |
| 6. 非調和modalから文化的調律体系が生じる | 非調和スペクトルに対応する局所協和の極小や、それに適合する音階候補は導ける。しかし極小の存在から、学習・継承・社会的選択を伴う文化的体系の成立は導けない。Setharesの原著は非調和音色と協和的音程の設計可能性を示す。ガムラン実測は楽器の非調和モード、製作者が調整した倍音成分、音階・知覚・様式的構造の併存を報告するが、非調和性単独の文化的因果を証明しない。[Sethares 1993 原著著者ページ](https://sethares.engr.wisc.edu/papers/consance.html)、[Carteretteほか 1993 原著](https://www.jstage.jst.go.jp/article/ast1980/14/6/14_6_383/_article/-char/en)。 |

F0監査のI11-1再取得は既存Sine／Harmonic／Modalの選択差とWAV差、および登録範囲の回帰を示す。自励発振、位相roughness知覚、pulseの経路別弁別、文化的調律生成を検証した証拠ではない。身体適応度計画も、非調和音程・音域ではH/Rとgravity等を、時間構造ではflux・meter・onset・arrivalを分ける方針を明記する。本メモの六点を公開技術説明や新実装の成立宣言へ転用しない。
